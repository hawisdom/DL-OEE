import logging
import os
import random
from sklearn.metrics import classification_report
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, matthews_corrcoef
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
from torch import nn
from collections import Counter

from datasets import my_collate
from transformers import AdamW
torch.set_printoptions(profile="full")

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def get_input_from_batch(batch):
    inputs = {  'tokens_ids': batch[0],
                'pos_class': batch[1], 
                'dep_ids': batch[2],
                'text_len': batch[3],
                'level': batch[4],
                'adj':batch[5],
                'adj_node_type':batch[6]
                }
    labels = batch[7]
    labels = labels.view(labels.shape[0]*labels.shape[1])
    return inputs, labels


def get_collate_fn():
    return my_collate

def train(args, train_dataset,train_labels_weight, model, test_dataset,test_labels_weight):
    '''Train the model'''
    tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size
    train_sampler = RandomSampler(train_dataset)
    collate_fn = get_collate_fn()
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=args.train_batch_size,
                                  collate_fn=collate_fn)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (
            len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(
            train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs


    parameters = filter(lambda param: param.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Train
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d",
                args.per_gpu_train_batch_size)
    logger.info("  Gradient Accumulation steps = %d",
                args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)


    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    all_eval_results = []
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    set_seed(args)
    epoch = 0
    modelpath = os.path.join(args.output_dir,
                             str(args.per_gpu_train_batch_size) + "_" + str(args.learning_rate) + ".model")
    output_eval_file = os.path.join(args.output_dir, 'eval_results.txt')
    writer = open(output_eval_file, 'a+')
    for _ in train_iterator:
        # epoch_iterator = tqdm(train_dataloader, desc='Iteration')
        for step, batch in enumerate(train_dataloader):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            inputs, labels = get_input_from_batch(batch)
            logit = model(**inputs)
            loss = F.cross_entropy(logit, labels, weight=train_labels_weight.to(args.device))
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                # scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                # Log metrics
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    results, eval_loss = evaluate(args, test_dataset,test_labels_weight, model,writer,epoch)
                    all_eval_results.append(results)
                    for key, value in results.items():
                        tb_writer.add_scalar(
                            'eval_{}'.format(key), value, global_step)
                    tb_writer.add_scalar('eval_loss', eval_loss, global_step)
                    tb_writer.add_scalar(
                        'train_loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss
            if step % 50 == 0:
                torch.save(model, modelpath)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            epoch_iterator.close()
            break
        epoch += 1
        tb_writer.add_scalar('train_epoch_loss',(tr_loss - logging_loss) / args.logging_steps, epoch)

    tb_writer.close()
    evaluate(args, test_dataset, test_labels_weight, model,writer,epoch)
    return global_step, tr_loss/global_step, all_eval_results


def evaluate(args, eval_dataset,test_labels_weight, model,writer,epoch):
    results = {}

    args.eval_batch_size = args.per_gpu_eval_batch_size
    eval_sampler = SequentialSampler(eval_dataset)
    collate_fn = get_collate_fn()
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
                                 batch_size=args.eval_batch_size,
                                 collate_fn=collate_fn)

    # Eval
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in eval_dataloader:
    # for batch in tqdm(eval_dataloader, desc='Evaluating'):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs, labels = get_input_from_batch(batch)

            logits = model(**inputs)
            tmp_eval_loss = F.cross_entropy(logits, labels,weight=test_labels_weight.to(args.device))

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = labels.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(
                out_label_ids, labels.detach().cpu().numpy(), axis=0)
        torch.cuda.empty_cache()

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=1)
    
    result = compute_metrics(preds, out_label_ids)
    results.update(result)

    target_names = ['none', 'sub', 'pred', 'obj']
    class_result = classification_report(y_true=out_label_ids,y_pred=preds, target_names=target_names, digits=4)
    micro_f1 = f1_score(y_true=out_label_ids, y_pred=preds, average='micro')

    logger.info('***** Eval results *****')
    logger.info("  eval loss: %s", str(eval_loss))
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))
        writer.write("  %s = %s\n" % (key, str(result[key])))
        writer.write(class_result)
        writer.write("micro_f1:%f\n" %micro_f1)
        writer.write("epoch:%d" %epoch)
        writer.write('\n')
    writer.write('\n')
    print(class_result)
    return results, eval_loss


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds, average='macro')
    return {
        "acc": acc,
        "macro_f1": f1,
    }


def compute_metrics(preds, labels):
    return acc_and_f1(preds, labels)
