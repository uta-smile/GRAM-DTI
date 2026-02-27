import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys
from tqdm import tqdm

from models import OptimizedFourModalContrastiveModel
from embedding_dataset import PrecomputedEmbeddingDataset, EmbeddingCollateFunction
from memory_optimized_dataset import MemoryOptimizedEmbeddingDataset, BatchedEmbeddingDataset


def train(model, train_loader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    total_contra_loss = 0
    total_itm_loss = 0
    total_ic50_loss = 0
    total_ic50_acc = 0
    ic50_samples = 0

    drop_stats = {'none': 0, 'text': 0, 'hta': 0, 'smiles': 0, 'protein': 0}
    anchor_stats = {'text': 0, 'hta': 0, 'smiles': 0, 'protein': 0}

    gradient_stats = {
        'text': [],
        'smiles': [],
        'hta': [],
        'protein': []
    }

    try:
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')

        for batch_idx, batch in enumerate(pbar):
            smiles_embeddings = batch['smiles_embeddings'].to(device)
            text_embeddings = batch['text_embeddings'].to(device)
            hta_embeddings = batch['hta_embeddings'].to(device)
            protein_embeddings = batch['protein_embeddings'].to(device)
            ic50_targets = batch['ic50_targets'].to(device)
            ic50_mask = batch['ic50_mask'].to(device)

            total_loss_batch, contra_loss, itm_loss, ic50_loss, ic50_acc, ic50_logits, drop_info = model(
                smiles_embeddings, text_embeddings, hta_embeddings, protein_embeddings,
                ic50_targets, ic50_mask
            )

            strategy_info = drop_info['strategy']

            dropped_modality = strategy_info['dropped_modality']
            drop_stats[dropped_modality] += 1

            if strategy_info['should_drop']:
                anchor_modality = strategy_info['anchor_modality']
                anchor_stats[anchor_modality] += 1

            if 'gradient_scores' in strategy_info and strategy_info['gradient_scores']:
                for modality, score in strategy_info['gradient_scores'].items():
                    gradient_stats[modality].append(score)

            optimizer.zero_grad()
            total_loss_batch.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += total_loss_batch.item()
            total_contra_loss += contra_loss.item()
            total_itm_loss += itm_loss.item()
            total_ic50_loss += ic50_loss.item() if isinstance(ic50_loss, torch.Tensor) else ic50_loss
            total_ic50_acc += ic50_acc.item() if isinstance(ic50_acc, torch.Tensor) else ic50_acc
            ic50_samples += ic50_mask.sum().item()

            strategy_info = drop_info['strategy']
            grad_display = ""
            if 'current_gradients' in strategy_info and strategy_info['current_gradients']:
                current_grads = strategy_info['current_gradients']
                grad_display = f"g_t:{current_grads.get('text', 0):.2f},g_s:{current_grads.get('smiles', 0):.2f}"

            dropped_display = strategy_info['dropped_modality'][:1] if strategy_info['should_drop'] else 'N'
            anchor_display = strategy_info['anchor_modality'][:1] if strategy_info['should_drop'] else 'N'

            pbar.set_postfix({
                'loss': f"{total_loss_batch.item():.3f}",
                'contra': f"{contra_loss.item():.3f}",
                'itm': f"{itm_loss.item():.3f}",
                'ic50_acc': f"{ic50_acc.item() if isinstance(ic50_acc, torch.Tensor) else ic50_acc:.3f}",
                'drop': dropped_display,
                'anchor': anchor_display,
                'grads': grad_display
            })

    except Exception as e:
        print(f"Error occurred during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    avg_loss = total_loss / len(train_loader)
    avg_contra_loss = total_contra_loss / len(train_loader)
    avg_itm_loss = total_itm_loss / len(train_loader)
    avg_ic50_loss = total_ic50_loss / len(train_loader) if ic50_samples > 0 else 0
    avg_ic50_acc = total_ic50_acc / len(train_loader) if ic50_samples > 0 else 0

    print(f'Epoch {epoch}: Average Loss = {avg_loss:.4f}')
    print(f'  Contrastive Loss = {avg_contra_loss:.4f}')
    print(f'  ITM Loss = {avg_itm_loss:.4f}')
    print(f'  IC50 Loss = {avg_ic50_loss:.4f}')
    print(f'  IC50 Accuracy = {avg_ic50_acc:.4f} (on {ic50_samples} samples)')

    total_drops = sum(drop_stats.values())
    print(f'  Consistent Drop Strategy Stats:')
    for modality, count in drop_stats.items():
        percentage = (count / total_drops * 100) if total_drops > 0 else 0
        print(f'    {modality}: {count} ({percentage:.1f}%)')

    total_anchors = sum(anchor_stats.values())
    if total_anchors > 0:
        print(f'  Anchor Modality Stats (when dropping):')
        for modality, count in anchor_stats.items():
            percentage = (count / total_anchors * 100) if total_anchors > 0 else 0
            print(f'    {modality}: {count} ({percentage:.1f}%)')

    print(f'  Average Gradient Scores:')
    for modality in ['text', 'smiles', 'hta', 'protein']:
        if gradient_stats[modality]:
            avg_grad = sum(gradient_stats[modality]) / len(gradient_stats[modality])
            print(f'    {modality}: {avg_grad:.4f}')
        else:
            print(f'    {modality}: N/A')

    print(f'  Drop Strategy: {model.gram_loss.gradient_drop_strategy}')
    print(f'  Gradient History Length: {len(model.gram_loss.gradient_history["text"])}')
    print(f'  Forward-Reverse Consistency: ENFORCED')

    return avg_loss

def main_worker(embedding_dir, batch_size, num_epochs, dataset_type='memory',
                use_multiprocessing=False, drop_prob=0.3, model_save_path='OPTIMIZED_EMBEDDINGS_gradient_adaptive_drop_modality',
                save_every_n_epochs=2, lambda_1=1.0, lambda_2=1.0, lambda_3=1.0, gradient_std_multiplier=1.5, gradient_history_length=5):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if dataset_type == 'memory':
        dataset = MemoryOptimizedEmbeddingDataset(embedding_dir, load_to_memory=True)
    elif dataset_type == 'lazy':
        dataset = MemoryOptimizedEmbeddingDataset(embedding_dir, load_to_memory=False)
    elif dataset_type == 'batched':
        dataset = BatchedEmbeddingDataset(embedding_dir, batch_cache_size=2000)
    else:
        dataset = PrecomputedEmbeddingDataset(embedding_dir)

    class_weights = None
    print(f"Dataset type: {dataset_type}")
    print(f"Dataset size: {len(dataset)} samples")
    print(f"Drop modality probability: {drop_prob}")
    print(f"Gradient-based modality dropping: Drop decisions based on gradient contributions")
    print(f"Forward-Reverse Consistency: ENFORCED (Single drop decision per forward pass)")
    print(f"Using single GPU training (avoiding NCCL/distributed issues)")
    class_weights = dataset.compute_class_weights()

    if class_weights is None:
        class_weights = [1.0, 1.0, 1.0]

    collate_fn = EmbeddingCollateFunction()

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )
    print("Using single-process DataLoader")

    model = OptimizedFourModalContrastiveModel(
        class_weights=class_weights,
        drop_prob=drop_prob,
        lambda_1=lambda_1,
        lambda_2=lambda_2,
        lambda_3=lambda_3,
        gradient_std_multiplier=gradient_std_multiplier,
        gradient_history_length=gradient_history_length
    ).to(device)

    trainable_params = list(model.parameters())

    total_trainable = sum(p.numel() for p in trainable_params)
    print(f"Total trainable parameters: {total_trainable:,}")

    optimizer = optim.Adam(trainable_params, lr=1e-4)

    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, device, epoch)

        if (epoch + 1) % save_every_n_epochs == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
                'class_weights': class_weights,
                'drop_prob': drop_prob,
                'lambda_1': lambda_1,
                'lambda_2': lambda_2,
                'lambda_3': lambda_3,
                'gradient_std_multiplier': gradient_std_multiplier,
                'gradient_history_length': gradient_history_length
            }
            os.makedirs(model_save_path, exist_ok=True)
            torch.save(checkpoint, f'{model_save_path}/4modal_optimized_consistent_drop_epoch_{epoch+1}.pt')

