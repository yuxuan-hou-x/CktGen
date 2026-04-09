"""Prediction evaluation utilities for surrogate/evaluator models.

Copyright 2025 CktGen Authors.

Licensed under the MIT License.
"""


import numpy as np
import torch

import utils.data as utils_data

from tqdm import tqdm


def _iter_graph_batches(graphs, batch_size):
    """Yield graph mini-batches for forward-only evaluation."""
    for start in range(0, len(graphs), batch_size):
        yield graphs[start:start + batch_size]


def _ensure_fom_stats(args, model, datasets):
    """Ensure FoM normalization statistics exist on the loaded evaluator."""
    if model.fom_train_mean is not None and model.fom_train_std is not None:
        return

    mean, std = utils_data.get_fom_train_mean_and_std(args, datasets['train'])
    model.set_train_mean_std(mean, std)


@torch.no_grad()
def evaluate_prediction(args, model, datasets):
    """Evaluate prediction quality and test loss on the full test set."""
    model.eval()
    _ensure_fom_stats(args, model, datasets)

    infer_batch_size = args.get('infer_batch_size', args.get('batch_size', 128))
    test_graphs = datasets['test']
    num_batches = (len(test_graphs) + infer_batch_size - 1) // infer_batch_size

    total_loss = 0.0
    total_nce = 0.0
    total_align = 0.0
    total_pred = 0.0

    total_gain_correct = 0
    total_bw_correct = 0
    total_pm_correct = 0
    pred_foms = []
    gnd_foms = []

    for batch_graphs in tqdm(
        _iter_graph_batches(test_graphs, infer_batch_size),
        total=num_batches,
        desc="Evaluating Prediction",
        unit="batch"
    ):
        batch_graphs = utils_data.collate_fn(batch_graphs)
        batch = utils_data.transforms(args, batch_graphs)
        gnd_fom = batch['foms'].detach().clone()

        batch['foms'] = utils_data.standard_fom(
            model.fom_train_mean,
            model.fom_train_std,
            batch['foms']
        )

        ckt_embs = model.get_ckt_embeddings(batch)
        spec_embs = model.get_spec_embeddings(batch)
        batch['ckt_embs'] = ckt_embs
        batch['spec_embs'] = spec_embs

        mixed_loss, losses = model.compute_loss(batch)
        total_loss += mixed_loss.item()
        total_nce += losses['nce'].item()
        total_align += losses['align'].item()
        total_pred += losses['pred'].item()

        preds = model.predict(ckt_embs, topk=1)
        pred_gain = preds['gain'].squeeze(-1)
        pred_bw = preds['bw'].squeeze(-1)
        pred_pm = preds['pm'].squeeze(-1)
        pred_fom = preds['fom'].reshape(-1)

        gain_correct = pred_gain.eq(batch['gains'])
        bw_correct = pred_bw.eq(batch['bws'])
        pm_correct = pred_pm.eq(batch['pms'])

        total_gain_correct += gain_correct.sum().item()
        total_bw_correct += bw_correct.sum().item()
        total_pm_correct += pm_correct.sum().item()

        pred_foms.append(pred_fom.detach().cpu())
        gnd_foms.append(gnd_fom.reshape(-1).detach().cpu())

    num_test = len(test_graphs)
    pred_foms = torch.cat(pred_foms, dim=0).numpy()
    gnd_foms = torch.cat(gnd_foms, dim=0).numpy()

    fom_mae = float(np.mean(np.abs(pred_foms - gnd_foms)))
    fom_rmse = float(np.sqrt(np.mean((pred_foms - gnd_foms) ** 2)))
    train_std = float(model.fom_train_std.detach().cpu().item())
    fom_nrmse_std = fom_rmse / max(train_std, 1e-8)

    return {
        'test_loss': total_loss / num_test,
        'test_nce_loss': total_nce / num_test,
        'test_align_loss': total_align / num_test,
        'test_pred_loss': total_pred / num_test,
        'gain_acc': total_gain_correct / num_test,
        'bw_acc': total_bw_correct / num_test,
        'pm_acc': total_pm_correct / num_test,
        'fom_mae': fom_mae,
        'fom_nrmse_std': fom_nrmse_std,
        'num_test_samples': num_test,
    }


@torch.no_grad()
def evaluate(args, model, datasets, logger):
    """Main evaluation entry point for trained evaluator models."""
    logger.info('###############################################################################')
    logger.info('                            Prediction Evaluation')
    logger.info('###############################################################################')

    res = evaluate_prediction(args, model, datasets)

    logger.info(
        'Test Loss: %.06f, nce: %.06f, align: %.06f, pred: %.06f'
        % (
            res['test_loss'],
            res['test_nce_loss'],
            res['test_align_loss'],
            res['test_pred_loss'],
        )
    )
    logger.info(
        'Gain Acc: %.06f, BW Acc: %.06f, PM Acc: %.06f, FoM MAE: %.06f, FoM NRMSE(std): %.06f'
        % (
            res['gain_acc'],
            res['bw_acc'],
            res['pm_acc'],
            res['fom_mae'],
            res['fom_nrmse_std'],
        )
    )
    logger.info(
        'Evaluator test samples: %d'
        % res['num_test_samples']
    )

    return res
