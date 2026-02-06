# -*- coding: utf-8 -*-
# @description: 
# @author: zchen
# @time: 2020/11/29 20:09
# @file: train.py
import logging
import os

import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler)
from tqdm import tqdm, trange
from transformers import AdamW, get_linear_schedule_with_warmup, BertConfig, BertTokenizer
import numpy as np
import conlleval as conlleval
from config import Config
from models import BERT_BiLSTM_CRF
from processor import NerProcessor
from loss import cross_entropy, entropy_loss, regression_loss

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def sample_selector(l1, l2, drop_rate):
    ind_sorted_1 = torch.argsort(l1.data)  # ascending order
    ind_sorted_2 = torch.argsort(l2.data)
    num_remember = max(int((1 - drop_rate) * l1.shape[0]), 1)
    ind_clean_1 = ind_sorted_1[:num_remember]
    ind_clean_2 = ind_sorted_2[:num_remember]
    ind_unclean_1 = ind_sorted_1[num_remember:]
    ind_unclean_2 = ind_sorted_2[num_remember:]
    return {'clean1': ind_clean_1, 'clean2': ind_clean_2, 'unclean1': ind_unclean_1, 'unclean2': ind_unclean_2}

def kl_div(p, q):
    """
    计算两个概率分布p和q之间的KL散度。

    :param p: 预测分布，通常是模型输出经过softmax后的结果。
    :param q: 目标分布，即混合标签分布。
    :return: KL散度损失。
    """
    return F.kl_div(p.log(), q, reduction='batchmean')

def create_mixed_labels(unclean_logits, clean_logits, alpha=0.5):
    """
    生成混合标签分布。

    :param unclean_logits: 不干净样本的预测logits。
    :param clean_logits: 干净样本的预测logits。
    :param alpha: 混合比例，默认为0.5。
    :return: 混合标签分布。
    """
    return alpha * F.softmax(unclean_logits, dim=1) + (1 - alpha) * F.softmax(clean_logits, dim=1)

def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def seqmix_data(inputs, labels, config):
    """
    生成混合的输入数据和标签用于SeqMix。
    
    参数:
    - inputs: 输入序列张量，形状为 (batch_size, seq_length)
    - labels: 标签张量，形状为 (batch_size, num_classes)
    - config: 配置对象，包含SeqMix所需的参数

    返回:
    - mixed_inputs: 混合后的输入序列
    - targets_a: 混合目标标签a
    - targets_b: 混合目标标签b
    - lam: 用于混合的lambda系数
    """
    
    # 从Beta分布中采样混合比例lambda
    alpha = config.alpha  # 默认值为0.4
    lam = np.random.beta(alpha, alpha)
    
    batch_size = inputs.size(0)
    if config.use_gpu:
        device = config.device
        index = torch.randperm(batch_size).to(device)
    else:
        index = torch.randperm(batch_size)
    
    # 生成混合输入
    mixed_inputs = lam * inputs + (1 - lam) * inputs[index, :]
    
    # 生成混合目标标签
    targets_a = labels
    targets_b = labels[index]
    
    return mixed_inputs, targets_a, targets_b, lam

def rbf_kernel(x, gamma_param=None):
    """
    使用 PyTorch 实现的 RBF 核函数。

    :param x: 输入张量，形状为 (n_samples, n_features)
    :param gamma: RBF 核的超参数，如果为 None，默认使用 1 / n_features
    :return: 计算得到的核矩阵，形状为 (n_samples, n_samples)
    """
    # 如果 gamma 未指定，则使用 1 / n_features 作为默认值
    if gamma_param is None:
        gamma_param = 1.0 / x.shape[1]

    # 计算 x 的 L2 范数平方 (x_norm)
    x_norm = np.sum(x ** 2, axis=-1)
    
    # 将 x_norm 进行 reshape，以便进行矩阵计算
    x_norm = x_norm.reshape(-1, 1)
    
    # 使用广播机制计算 RBF 核矩阵
    kernel_matrix = np.exp(-gamma_param * (x_norm + x_norm.T - 2.0 * np.dot(x, x.T)))

    return kernel_matrix


def train():
    """
    模型训练
    :return:
    """
    processor = NerProcessor()
    config = Config()

    # 清理output/xxx目录，若output/xxx目录存在，将会被删除, 然后初始化输出目录
    processor.clean_output(config)

    # SummaryWriter构造函数
    writer = SummaryWriter(logdir=os.path.join(config.output_path, "eval"), comment="ner")

    # 如果显存不足，我们可以通过gradient_accumulation_steps梯度累计来解决
    # 假设原来的batch_size = 10, 数据总量为1000，那么一共需要100train_steps，同时一共进行100次梯度更新。
    # 若是显存不够，我们需要减小batch＿size，我们设置gradient_accumulation_steps = 2，设置batch＿size = 5，
    # 我们需要运行两次，才能在内存中放入10条数据，梯度更新的次数不变为100次，那么我们的train＿steps = 200
    if config.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            config.gradient_accumulation_steps))

    # 配置可用设备，没有指定使用哪一块gpu，则全部使用
    use_gpu = torch.cuda.is_available() and config.use_gpu
    device = torch.device('cuda' if use_gpu else config.device)
    config.device = device
    n_gpu = torch.cuda.device_count()
    logger.info(f"available device: {device}，count_gpu: {n_gpu}")

    logger.info("====================== Start Data Pre-processing ======================")

    # 读取训练数据获取标签
    label_list = processor.get_labels(config=config)
    print(label_list)
    config.label_list = label_list
    num_labels = len(label_list)
    logger.info(f"loading labels successful! the size is {num_labels}, label is: {','.join(list(label_list))}")

    # 获取label2id、id2label的映射
    label2id, id2label = processor.get_label2id_id2label(config.output_path, label_list=label_list)
    logger.info("loading label2id and id2label dictionary successful!")

    if config.do_train:
        # 初始化tokenizer(标记生成器)、bert_config、BERT_BiLSTM_CRF
        tokenizer = BertTokenizer.from_pretrained(config.model_name_or_path, do_lower_case=config.do_lower_case)
        bert_config = BertConfig.from_pretrained(config.model_name_or_path, num_labels=num_labels)
          # 加载两个模型
        model1 = BERT_BiLSTM_CRF.from_pretrained(config.model_name_or_path, config=bert_config,
                                                 need_birnn=config.need_birnn, rnn_dim=config.rnn_dim)
        model2 = BERT_BiLSTM_CRF.from_pretrained(config.model_name_or_path, config=bert_config,
                                                 need_birnn=config.need_birnn, rnn_dim=config.rnn_dim)
        model1.to(device)
        model2.to(device)
        logger.info("loading tokenizer、bert_config and bert_bilstm_crf model successful!")

        if use_gpu and n_gpu > 1:
            model1 = torch.nn.DataParallel(model1)
            model2 = torch.nn.DataParallel(model2)

        logger.info("starting load train data and data_loader...")
        # 获取训练样本、样本特征、TensorDataset信息
        train_examples, train_features, train_data = processor.get_dataset(config, tokenizer,
                                                                           mode="train")
        # 训练数据载入
        train_data_loader = DataLoader(train_data, batch_size=config.train_batch_size,
                                       sampler=RandomSampler(train_data))
        logger.info("loading train data_set and data_loader successful!")

        eval_examples, eval_features, eval_data = [], [], None
        if config.do_eval:
            logger.info("starting load eval data...")
            eval_examples, eval_features, eval_data = processor.get_dataset(config, tokenizer,
                                                                            mode="eval")
            logger.info("loading eval data_set successful!")
        logger.info("====================== End Data Pre-proces sing ======================")

        # 初始化模型参数优化器
        no_decay = ['bias', 'LayerNorm.weight']
   
        optimizer1 = initialize_optimizer(model1, config)
        optimizer2 = initialize_optimizer(model2, config)

        # 初始化学习率优化器
        t_total = len(train_data_loader) // config.gradient_accumulation_steps * config.num_train_epochs
        scheduler1 = get_linear_schedule_with_warmup(optimizer1, num_warmup_steps=config.warmup_steps, num_training_steps=t_total)
        scheduler2 = get_linear_schedule_with_warmup(optimizer2, num_warmup_steps=config.warmup_steps, num_training_steps=t_total)

        # scheduler = WarmupLinearSchedule(optimizer, warmup_steps=config.warmup_steps, t_total=t_total)
        logger.info("loading AdamW optimizer、Warmup LinearSchedule and calculate optimizer parameter successful!")

        logger.info("====================== Running training ======================")
        logger.info(
            f"Num Examples:  {len(train_data)}, Num Batch Step: {len(train_data_loader)}, "
            f"Num Epochs: {config.num_train_epochs}, Num scheduler steps：{t_total}")

        # 启用 BatchNormalization 和 Dropout
        global_step, tr_loss1,tr_loss2, logger_loss, best_f1 = 0, 0.0,0.0, 0.0, 0.0

         # 阶段2所需的调度器
        drop_rate_scheduler = np.linspace(config.drop_rate, config.final_drop_rate, num=config.epochs)

        for ep in trange(int(config.num_train_epochs), desc="Epoch"):
            logger.info(f"########[Epoch: {ep}/{int(config.num_train_epochs)}]########")
            model1.train()
            model2.train()

            for step, batch in enumerate(tqdm(train_data_loader, desc="DataLoader")):
                logger.info(f"####[Step: {step}/{len(train_data_loader)}]####")
                batch = tuple(t.to(device) for t in batch)
                input_ids, token_type_ids, attention_mask, label_ids = batch

                 # 第一步：前向传播，获取输出
                outputs1 = model1(input_ids, label_ids, token_type_ids, attention_mask)
                outputs2 = model2(input_ids, label_ids, token_type_ids, attention_mask)
                #真的就只返回loss
                loss1 = outputs1[0]
                loss2 = outputs2[0]
                #获得logits值
                logits1 = outputs1[1]
                logits2 = outputs2[1]
                #print("=========",logits1.shape)
                # 第二步：根据训练阶段调整训练逻辑

                if ep < config.stage1:  # 阶段1：暖启动阶段
                    loss1 = cross_entropy(logits1, label_ids)
                    loss2 = cross_entropy(logits2, label_ids)

                else:  # 阶段2：协同教学阶段
                    # 获取样本选择
                    with torch.no_grad():
                        #根据模型的交叉熵损失来选择干净的样本和不干净的样本
                        cce_losses1 = cross_entropy(logits1, label_ids, reduction='none')
                        cce_losses2 = cross_entropy(logits2, label_ids, reduction='none')
                        # 保存了被标记为干净和不干净的样本索引。这些索引用于在后续步骤中对不同类别的样本进行处理。
                        sample_selection = sample_selector(cce_losses1, cce_losses2, drop_rate_scheduler[ep])

                    clean_logits1 = logits1[sample_selection['clean2']]
                    clean_logits2 = logits2[sample_selection['clean1']]
                    clean_labels1 = label_ids[sample_selection['clean2']]
                    clean_labels2 = label_ids[sample_selection['clean1']]

                    loss_c1 = cross_entropy(clean_logits1, clean_labels1) + entropy_loss(clean_logits1)
                    loss_c2 = cross_entropy(clean_logits2, clean_labels2) + entropy_loss(clean_logits2)

                    
                    #不干净样本实际需要通过核函数划分出分布外噪声
                    unclean_logits1 = logits1[sample_selection['unclean2']]
                    unclean_logits2 = logits2[sample_selection['unclean1']]

                    #使用核函数。rbf_kernel 函数计算样本之间的相似度，生成一个相似度矩阵。然后，可以根据相似度阈值来筛选噪声样本
                    kernel_matrix1 = rbf_kernel(unclean_logits1.cpu().detach().numpy())
                    kernel_matrix2 = rbf_kernel(unclean_logits2.cpu().detach().numpy())

                    # 基于相似度阈值过滤噪声样本（需要微调）
                    threshold = 0.5
                    mask1 = np.mean(kernel_matrix1, axis=1) > threshold
                    mask2 = np.mean(kernel_matrix2, axis=1) > threshold

                    filtered_logits1 = unclean_logits1[mask1]
                    filtered_logits2 = unclean_logits2[mask2]


                    #继续不干净样本训练（使用KL散度和混合标签分布）主要用作分布外噪声矫正
                    mixed_labels1 = create_mixed_labels(filtered_logits1, clean_logits1)
                    mixed_labels2 = create_mixed_labels(filtered_logits2, clean_logits2)

                    loss_u1 = kl_div(F.softmax(filtered_logits1, dim=1) + 1e-8, mixed_labels1).mean()
                    loss_u2 = kl_div(F.softmax(filtered_logits2, dim=1) + 1e-8, mixed_labels2).mean()

                    # Mixup
                    """
                    SeqMix 数据混合用作数据增强。
                    SeqMix 的核心在于生成混合数据和混合标签。这里，seqmix_data 函数通过线性插值（基于 lambda 权重）
                    在嵌入空间中混合不干净样本的输入序列和标签，从而生成新的样本对。
                    """      
                    mixed_inputs1, targets_a1, targets_b1, lam1 = seqmix_data(input_ids[sample_selection['unclean2']], mixed_labels1, config)
                    mixed_inputs2, targets_a2, targets_b2, lam2 = seqmix_data(input_ids[sample_selection['unclean1']], mixed_labels2, config)
     
                    mixed_inputs1 = mixed_inputs1.long()
                    mixed_inputs2 = mixed_inputs2.long()

                    # 注意：计算 logits_mixed 后，需要将其传递给模型的前向传播函数，并正确传递输入
                    logits_mixed1 = model1(mixed_inputs1, token_type_ids[sample_selection['unclean2']], attention_mask[sample_selection['unclean2']])[1]
                    logits_mixed2 = model2(mixed_inputs2, token_type_ids[sample_selection['unclean1']], attention_mask[sample_selection['unclean1']])[1]

                    # 将 logits 转换为概率分布
                    probs_mixed1 = F.softmax(logits_mixed1, dim=1)
                    probs_mixed2 = F.softmax(logits_mixed2, dim=1)

                    # 混合目标分布
                    mixed_targets1 = lam1 * targets_a1 + (1 - lam1) * targets_b1
                    mixed_targets2 = lam2 * targets_a2 + (1 - lam2) * targets_b2
                    # 计算 KL 散度损失
                    # 假设 kl_div 已经计算出 KL 散度损失
                    kl_div_loss1 = kl_div(probs_mixed1, mixed_targets1)
                    kl_div_loss2 = kl_div(probs_mixed2, mixed_targets2)

                    # 在 mixup_criterion 中直接使用 kl_div_loss1 和 kl_div_loss2
                    loss_n1 = lam1 * kl_div_loss1 + (1 - lam1) * kl_div_loss1
                    loss_n2 = lam2 * kl_div_loss2 + (1 - lam2) * kl_div_loss2

                    # 最终损失计算
                    loss1 = loss_c1 + config.beta * loss_u1+ config.beta * loss_n1
                    loss2 = loss_c2 + config.beta * loss_u2+ config.beta * loss_n2

                if use_gpu and n_gpu > 1:
                    # mean() to average on multi-gpu.
                    loss1 = loss1.mean()
                    loss2 = loss2.mean()
                if config.gradient_accumulation_steps > 1:
                    loss1 = loss1 / config.gradient_accumulation_steps
                    loss2 = loss2 / config.gradient_accumulation_steps

                # 反向传播
                loss1.backward()
                loss2.backward()
                tr_loss1 += loss1.item()
                tr_loss2 += loss2.item()

                # 优化器_模型参数的总更新次数，和上面的t_total对应
                if (step + 1) % config.gradient_accumulation_steps == 0:
                    
                    optimizer1.step()
                    optimizer2.step()
                    scheduler1.step()
                    scheduler2.step()
                    model1.zero_grad()
                    model2.zero_grad()
                    global_step += 1

                    if config.logging_steps > 0 and global_step % config.logging_steps == 0:
                        tr_loss_avg1 = (tr_loss1 - tr_loss1) / config.logging_steps
                        tr_loss_avg2 = (tr_loss2 - tr_loss2) / config.logging_steps
                        writer.add_scalar("Train/loss_model1", tr_loss_avg1, global_step)
                        writer.add_scalar("Train/loss_model2", tr_loss_avg2, global_step)

            # 模型验证
            if config.do_eval:
                overall1, by_type1 = evaluate(config, eval_data, model1, id2label, [f.ori_tokens for f in eval_features])
                overall2, by_type2 = evaluate(config, eval_data, model2, id2label, [f.ori_tokens for f in eval_features])

                f1_score1 = overall1.fscore
                f1_score2 = overall2.fscore

                writer.add_scalar("Eval/precision_model1", overall1.prec, ep)
                writer.add_scalar("Eval/precision_model2", overall2.prec, ep)
                writer.add_scalar("Eval/recall_model1", overall1.rec, ep)
                writer.add_scalar("Eval/recall_model2", overall2.rec, ep)
                writer.add_scalar("Eval/f1_score_model1", f1_score1, ep)
                writer.add_scalar("Eval/f1_score_model2", f1_score2, ep)

                # 保存表现最佳的模型
                if f1_score1 > best_f1:
                    logger.info(f"******** the best f1 for model1 is {f1_score1}, save model !!! ********")
                    best_f1 = f1_score1
                    save_model(config, model1, tokenizer)

                if f1_score2 > best_f1:
                    logger.info(f"******** the best f1 for model2 is {f1_score2}, save model !!! ********")
                    best_f1 = f1_score2
                    save_model(config, model2, tokenizer)

        writer.close()
        logger.info("NER model training successful!!!")

    if config.do_test:
        tokenizer = BertTokenizer.from_pretrained(config.output_path, do_lower_case=config.do_lower_case)
        config = torch.load(os.path.join(config.output_path, 'training_config.bin'))
        model1 = BERT_BiLSTM_CRF.from_pretrained(config.output_path, need_birnn=config.need_birnn,
                                             rnn_dim=config.rnn_dim)
        model1.to(device)

        # 加载第二个模型
        model2 = BERT_BiLSTM_CRF.from_pretrained(config.output_path, need_birnn=config.need_birnn,
                                                rnn_dim=config.rnn_dim)
        model2.to(device)

        test_examples, test_features, test_data = processor.get_dataset(config, tokenizer, mode="test")

        logger.info("====================== Running test ======================")
        logger.info(f"Num Examples:  {len(test_examples)}, Batch size: {config.eval_batch_size}")

        all_ori_tokens = [f.ori_tokens for f in test_features]
        all_ori_labels = [e.label.split(" ") for e in test_examples]
        test_sampler = SequentialSampler(test_data)
        test_data_loader = DataLoader(test_data, sampler=test_sampler, batch_size=config.eval_batch_size)
        
        model1.eval()
        model2.eval()


        pred_labels = []
        for b_i, (input_ids, token_type_ids, attention_mask, label_ids) in enumerate(
            tqdm(test_data_loader, desc="TestDataLoader")):

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)

            with torch.no_grad():
                # 使用模型1进行预测
                logits1 = model1.predict(input_ids, token_type_ids, attention_mask)
                # 使用模型2进行预测
                logits2 = model2.predict(input_ids, token_type_ids, attention_mask)

                # 将两个模型的输出结果进行组合（这里以平均为例）
                combined_logits = (logits1 + logits2) / 2

                # 将组合后的结果转换为标签
                for l in combined_logits:
                    pred_label = []
                    for idx in l:
                        pred_label.append(id2label[idx])
                    pred_labels.append(pred_label)

        assert len(pred_labels) == len(all_ori_tokens) == len(all_ori_labels)
        
        with open(os.path.join(config.output_path, "token_labels_test.txt"), "w", encoding="utf-8") as f:
            for ori_tokens, ori_labels, prel in zip(all_ori_tokens, all_ori_labels, pred_labels):
                for ot, ol, pl in zip(ori_tokens, ori_labels, prel):
                    if ot in ["[CLS]", "[SEP]"]:
                        continue
                    else:
                        f.write(f"{ot} {ol} {pl}\n")
                f.write("\n")


def evaluate(config: Config, data, model, id2label, all_ori_tokens):
    """
    
    :param config:
    :param data:
    :param model:
    :param id2label:
    :param all_ori_tokens:
    :return:
    """
    ori_labels, pred_labels = [], []
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model.eval()
    sampler = SequentialSampler(data)
    data_loader = DataLoader(data, sampler=sampler, batch_size=config.train_batch_size)
    for b_i, (input_ids, token_type_ids, attention_mask, label_ids) in enumerate(
            tqdm(data_loader, desc="Evaluating")):
        input_ids = input_ids.to(config.device)
        attention_mask = attention_mask.to(config.device)
        token_type_ids = token_type_ids.to(config.device)
        label_ids = label_ids.to(config.device)
        with torch.no_grad():
            logits = model.predict(input_ids, token_type_ids, attention_mask)

        for l in logits:
            pred_labels.append([id2label[idx] for idx in l])

        for l in label_ids:
            ori_labels.append([id2label[idx.item()] for idx in l])

    eval_list = []
    for ori_tokens, oril, prel in zip(all_ori_tokens, ori_labels, pred_labels):
        for ot, ol, pl in zip(ori_tokens, oril, prel):
            if ot in ["[CLS]", "[SEP]"]:
                continue
            eval_list.append(f"{ot} {ol} {pl}\n")
        eval_list.append("\n")

    # eval the model
    counts = conlleval.evaluate(eval_list)
    conlleval.report(counts)

    # namedtuple('Metrics', 'tp fp fn prec rec fscore')
    overall, by_type = conlleval.metrics(counts)
    return overall, by_type

def initialize_optimizer(model, config):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate, eps=config.adam_epsilon)
    return optimizer

def save_model(config, model, tokenizer):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(config.output_path)
    tokenizer.save_pretrained(config.output_path)
    torch.save(config, os.path.join(config.output_path, 'training_config.bin'))
    torch.save(model, os.path.join(config.output_path, 'ner_model.ckpt'))
    logger.info("Model and config saved successfully.")


if __name__ == '__main__':
    train()
