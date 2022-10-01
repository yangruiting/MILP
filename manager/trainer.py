import statistics
import timeit
import os
import logging
import pdb
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn import metrics
from copy import deepcopy


class Trainer(nn.Module):
    # graph_classifier
    def __init__(self, params, graph_classifier, train, valid):
        super(Trainer, self).__init__()
        self.graph_classifier = graph_classifier
        self.params = params
        self.train_data = train
        self.valid_data = valid
        self.updates_counter = 0
        self.update_lr = params.update_lr
        self.model_params = self.graph_classifier.parameters()
        logging.info('Total number of parameters: %d' % sum(map(lambda x: x.numel(), self.model_params)))

        if params.optimizer == "SGD":
            self.optimizer = optim.SGD(self.model_params, lr=params.lr, momentum=params.momentum,
                                       weight_decay=self.params.l2)
        if params.optimizer == "Adam":
            self.optimizer = optim.Adam(self.model_params, lr=params.lr, weight_decay=self.params.l2)

        self.criterion = nn.MarginRankingLoss(self.params.margin, reduction='sum')
        self.reset_training_state()

    def reset_training_state(self):
        self.best_metric = 0
        self.last_metric = 0
        self.not_improved_count = 0

    def train_epoch(self, i):
        total_loss = 0
        all_preds = []
        all_labels = []
        all_scores = []

        dataloader = DataLoader(self.train_data[i], batch_size=self.params.batch_size, shuffle=True, num_workers=0,
                                collate_fn=self.params.collate_fn)
        valid_dataloader = DataLoader(self.valid_data[i], batch_size=self.params.batch_size, shuffle=True,
                                      num_workers=0,
                                      collate_fn=self.params.collate_fn)
        self.graph_classifier.train()

        for b_idx, batch in enumerate(dataloader):
            (graph_pos, r_labels_pos), g_labels_pos, (graph_neg, r_labels_neg), g_labels_neg = batch

            g_labels_pos = torch.LongTensor(g_labels_pos).to(device=self.params.device)
            r_labels_pos = torch.LongTensor(r_labels_pos).to(device=self.params.device)
            g_labels_neg = torch.LongTensor(g_labels_neg).to(device=self.params.device)
            r_labels_neg = torch.LongTensor(r_labels_neg).to(device=self.params.device)

            self.graph_classifier.train()
            self.optimizer.zero_grad()
            self.updates_counter += 1

            if b_idx == 0:
                score_pos = self.graph_classifier(graph_pos, self.graph_classifier.parameters())
                score_neg = self.graph_classifier(graph_neg, self.graph_classifier.parameters())
                loss = self.criterion(score_pos.mean(dim=1), score_neg.view(len(score_pos), -1).mean(dim=1),
                                      torch.Tensor([1]).to(device=self.params.device))
                grad = torch.autograd.grad(loss, self.graph_classifier.parameters())
                fast_weights = list(
                    map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.graph_classifier.parameters())))

                with torch.no_grad():

                    for valid_b_idx, valid_batch in enumerate(valid_dataloader):
                        if valid_b_idx == 0:
                            (v_graphs_pos, v_r_labels_pos), v_g_labels_pos, (
                                v_graph_neg, v_r_labels_neg), v_g_labels_neg = valid_batch
                        break

                    v_g_labels_pos = torch.LongTensor(v_g_labels_pos).to(device=self.params.device)
                    v_r_labels_pos = torch.LongTensor(v_r_labels_pos).to(device=self.params.device)
                    v_g_labels_neg = torch.LongTensor(v_g_labels_neg).to(device=self.params.device)
                    v_r_labels_neg = torch.LongTensor(v_r_labels_neg).to(device=self.params.device)

                    valid_score_pos = self.graph_classifier(v_graphs_pos, self.graph_classifier.parameters())
                    valid_score_neg = self.graph_classifier(v_graph_neg, self.graph_classifier.parameters())
                    valid_loss = self.criterion(valid_score_pos.mean(dim=1),
                                                valid_score_neg.view(len(valid_score_pos), -1).mean(dim=1),
                                                torch.Tensor([1]).to(device=self.params.device))

                    all_scores += valid_score_pos.squeeze().detach().cpu().tolist() + valid_score_neg.squeeze().detach().cpu().tolist()
                    all_labels += v_g_labels_pos.tolist() + v_g_labels_neg.tolist()
                    total_loss += valid_loss
                with torch.no_grad():
                    valid_score_pos = self.graph_classifier(v_graphs_pos, fast_weights)
                    valid_score_neg = self.graph_classifier(v_graphs_pos, fast_weights)
                    # valid_score_neg = self.graph_classifier(v_graph_neg, fast_weights)
                    valid_loss = self.criterion(valid_score_pos.mean(dim=1),
                                                valid_score_neg.view(len(valid_score_pos), -1).mean(dim=1),
                                                torch.Tensor([1]).to(device=self.params.device))

                    all_scores += valid_score_pos.squeeze().detach().cpu().tolist() + valid_score_neg.squeeze().detach().cpu().tolist()
                    all_labels += v_g_labels_pos.tolist() + v_g_labels_neg.tolist()
                    total_loss += valid_loss
            else:
                score_pos = self.graph_classifier(graph_pos, fast_weights)
                score_neg = self.graph_classifier(graph_neg, fast_weights)
                loss = self.criterion(score_pos.mean(dim=1), score_neg.view(len(score_pos), -1).mean(dim=1),
                                      torch.Tensor([1]).to(device=self.params.device))

                grad = torch.autograd.grad(loss, self.graph_classifier.parameters())
                fast_weights = list(
                    map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.graph_classifier.parameters())))

                for b_idx, batch in enumerate(valid_dataloader):
                    (v_graphs_pos, v_r_labels_pos), v_g_labels_pos, (
                        v_graph_neg, v_r_labels_neg), v_g_labels_neg = batch
                    break

                valid_score_pos = self.graph_classifier(v_graphs_pos, fast_weights)
                valid_score_neg = self.graph_classifier(v_graph_neg, fast_weights)
                valid_loss = self.criterion(valid_score_pos.mean(dim=1),
                                            valid_score_neg.view(len(valid_score_pos), -1).mean(dim=1),
                                            torch.Tensor([1]).to(device=self.params.device))
                total_loss += valid_loss
                with torch.no_grad():
                    v_g_labels_pos = torch.LongTensor(v_g_labels_pos).to(device=self.params.device)
                    v_r_labels_pos = torch.LongTensor(v_r_labels_pos).to(device=self.params.device)
                    v_g_labels_neg = torch.LongTensor(v_g_labels_neg).to(device=self.params.device)
                    v_r_labels_neg = torch.LongTensor(v_r_labels_neg).to(device=self.params.device)
                    all_scores += valid_score_pos.squeeze().detach().cpu().tolist() + valid_score_neg.squeeze().detach().cpu().tolist()
                    all_labels += v_g_labels_pos.tolist() + v_g_labels_neg.tolist()

            auc = metrics.roc_auc_score(all_labels, all_scores)
            auc_pr = metrics.average_precision_score(all_labels, all_scores)
            weight_norm = sum(map(lambda x: torch.norm(x), self.model_params))

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss, auc, auc_pr, weight_norm

    def forward(self):
        self.reset_training_state()
        for epoch in range(1, self.params.num_epochs + 1):
            time_start = time.time()
            total_l = [0 for _ in range(4)]
            total_a = [0 for _ in range(4)]
            total_a_p = [0 for _ in range(4)]
            total_weight = [0 for _ in range(4)]
            for i in range(4):
                loss, auc, auc_pr, weight_norm = self.train_epoch(i)
                # print(loss)
                total_l[i] += loss
                total_a[i] += auc
                total_a_p[i] += auc_pr
                total_weight[i] += weight_norm

            total_loss = sum(total_l)
            total_auc = sum(total_a)
            total_auc_pr = sum(total_a_p)
            total_weight_norm = sum(total_weight)

            if self.params.eval_every_iter and self.updates_counter % self.params.eval_every_iter == 0:
                tic = time.time()
                result = total_auc / 4
                if result >= self.best_metric:
                    torch.save(self.graph_classifier, os.path.join(self.params.exp_dir,
                                                                   'best_graph_classifier.pth'))
                    logging.info('Better models found w.r.t accuracy. Saved it!')
                    self.best_metric = result
                    self.not_improved_count = 0
                else:
                    self.not_improved_count += 1
                    if self.not_improved_count > self.params.early_stop:
                        logging.info(
                            f"QuerySet performance didn\'t improve for {self.params.early_stop} epochs. Training stops.")
                        break
                self.last_metric = result

            time_elapsed = time.time() - time_start
            logging.info(
                f'Epoch {epoch} with loss: {total_loss / 4}, training auc: {total_auc / 4}, training auc_pr: {total_auc_pr / 4}, best QuerySet AUC: {self.best_metric}, weight_norm: {total_weight_norm / 4} in {time_elapsed}')
            # if epoch % self.params.save_every == 0:
            #     torch.save(self.graph_classifier,
            #                os.path.join(self.params.exp_dir, str(epoch) + 'graph_classifier_chk.pth'))
            if epoch == self.params.num_epochs:
                torch.save(self.graph_classifier, os.path.join(self.params.exp_dir,
                                                               'last_graph_classifier.pth'))
                logging.info('Last models found w.r.t accuracy. Saved it!')

    def finetuning(self):
        total_loss = 0
        all_preds = []
        all_labels = []
        all_scores = []
        net = deepcopy(self.graph_classifier)

        dataloader = DataLoader(self.train_data, batch_size=self.params.batch_size, shuffle=True, num_workers=0,
                                collate_fn=self.params.collate_fn)
        valid_dataloader = DataLoader(self.valid_data, batch_size=self.params.batch_size, shuffle=True,
                                      num_workers=0,
                                      collate_fn=self.params.collate_fn)
        self.graph_classifier.train()

        for b_idx, batch in enumerate(dataloader):
            (graph_pos, r_labels_pos), g_labels_pos, (graph_neg, r_labels_neg), g_labels_neg = batch
            g_labels_pos = torch.LongTensor(g_labels_pos).to(device=self.params.device)
            r_labels_pos = torch.LongTensor(r_labels_pos).to(device=self.params.device)
            g_labels_neg = torch.LongTensor(g_labels_neg).to(device=self.params.device)
            r_labels_neg = torch.LongTensor(r_labels_neg).to(device=self.params.device)

            self.graph_classifier.train()
            self.updates_counter += 1

            if b_idx == 0:
                score_pos = self.graph_classifier(graph_pos, self.graph_classifier.parameters())
                score_neg = self.graph_classifier(graph_neg, self.graph_classifier.parameters())
                loss = self.criterion(score_pos.mean(dim=1), score_neg.view(len(score_pos), -1).mean(dim=1),
                                      torch.Tensor([1]).to(device=self.params.device))
                grad = torch.autograd.grad(loss, self.graph_classifier.parameters())
                fast_weights = list(
                    map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.graph_classifier.parameters())))
                with torch.no_grad():
                    (v_graphs_pos, v_r_labels_pos), v_g_labels_pos, (v_graph_neg, v_r_labels_neg), v_g_labels_neg = \
                        next(iter(valid_dataloader))

                    v_g_labels_pos = torch.LongTensor(v_g_labels_pos).to(device=self.params.device)
                    v_r_labels_pos = torch.LongTensor(v_r_labels_pos).to(device=self.params.device)
                    v_g_labels_neg = torch.LongTensor(v_g_labels_neg).to(device=self.params.device)
                    v_r_labels_neg = torch.LongTensor(v_r_labels_neg).to(device=self.params.device)
                    valid_score_pos = self.graph_classifier(v_graphs_pos, self.graph_classifier.parameters())
                    valid_score_neg = self.graph_classifier(v_graph_neg, self.graph_classifier.parameters())
                    valid_loss = self.criterion(valid_score_pos.mean(dim=1),
                                                valid_score_neg.view(len(valid_score_pos), -1).mean(dim=1),
                                                torch.Tensor([1]).to(device=self.params.device))

                    all_scores += valid_score_pos.squeeze().detach().cpu().tolist() + valid_score_neg.squeeze().detach().cpu().tolist()
                    all_labels += v_g_labels_pos.tolist() + v_g_labels_neg.tolist()
                    total_loss += valid_loss
                with torch.no_grad():
                    valid_score_pos = self.graph_classifier(v_graphs_pos, fast_weights)
                    valid_score_neg = self.graph_classifier(v_graph_neg, fast_weights)
                    valid_loss = self.criterion(valid_score_pos.mean(dim=1),
                                                valid_score_neg.view(len(valid_score_pos), -1).mean(dim=1),
                                                torch.Tensor([1]).to(device=self.params.device))

                    all_scores += valid_score_pos.squeeze().detach().cpu().tolist() + valid_score_neg.squeeze().detach().cpu().tolist()
                    all_labels += v_g_labels_pos.tolist() + v_g_labels_neg.tolist()
                    total_loss += valid_loss
            else:
                score_pos = self.graph_classifier(graph_pos, fast_weights)
                score_neg = self.graph_classifier(graph_neg, fast_weights)
                loss = self.criterion(score_pos.mean(dim=1), score_neg.view(len(score_pos), -1).mean(dim=1),
                                      torch.Tensor([1]).to(device=self.params.device))
                grad = torch.autograd.grad(loss, self.graph_classifier.parameters())
                fast_weights = list(
                    map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.graph_classifier.parameters())))

                (v_graphs_pos, v_r_labels_pos), v_g_labels_pos, (v_graph_neg, v_r_labels_neg), v_g_labels_neg = \
                    next(iter(valid_dataloader))

                v_g_labels_pos = torch.LongTensor(v_g_labels_pos).to(device=self.params.device)
                v_r_labels_pos = torch.LongTensor(v_r_labels_pos).to(device=self.params.device)
                v_g_labels_neg = torch.LongTensor(v_g_labels_neg).to(device=self.params.device)
                v_r_labels_neg = torch.LongTensor(v_r_labels_neg).to(device=self.params.device)

                valid_score_pos = self.graph_classifier(v_graphs_pos, fast_weights)
                valid_score_neg = self.graph_classifier(v_graph_neg, fast_weights)
                valid_loss = self.criterion(valid_score_pos.mean(dim=1),
                                            valid_score_neg.view(len(valid_score_pos), -1).mean(dim=1),
                                            torch.Tensor([1]).to(device=self.params.device))
                all_scores += valid_score_pos.squeeze().detach().cpu().tolist() + valid_score_neg.squeeze().detach().cpu().tolist()
                all_labels += v_g_labels_pos.tolist() + v_g_labels_neg.tolist()
                total_loss += valid_loss

            auc = metrics.roc_auc_score(all_labels, all_scores)
            auc_pr = metrics.average_precision_score(all_labels, all_scores)

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        # torch.save(self.graph_classifier, os.path.join(self.params.exp_dir,
        #                                                'fine_graph_classifier.pth'))

        return {'auc': auc, 'auc_pr': auc_pr}