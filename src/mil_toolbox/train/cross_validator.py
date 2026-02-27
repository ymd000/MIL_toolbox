import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from mil_toolbox.data import FoldManager


class CrossValidationTrainer:
    def __init__(
        self,
        model_class,
        model_kwargs: dict,
        dataset,
        num_fold: int,
        output_dir: str = "./outputs",
        num_workers: int = 0,  # avoid copy overhead between subprocesses
        shuffle: bool = True,
        random_state: int = 42
    ):
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.dataset = dataset
        self.num_fold = num_fold
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.random_state = random_state

        # FoldManager初期化
        self.fold_manager = FoldManager(output_dir)

    def run(self):
        # fold作成・保存
        self.fold_manager.create_folds(
            self.dataset,
            self.num_fold,
            self.shuffle,
            self.random_state
        )
        self.fold_manager.save()

        results = []
        for fold_info in self.fold_manager.folds:
            print(f"\n=== Fold {fold_info.fold_idx + 1}/{self.num_fold} ===")
            fold_result = self.run_one_fold(fold_info)
            results.append(fold_result)
        return results

    def run_one_fold(self, fold_info):
        fold_idx = fold_info.fold_idx
        train_idx = fold_info.train_indices
        val_idx = fold_info.val_indices

        # 各foldでモデルを新規作成（重みを初期化）
        model = self.model_class(**self.model_kwargs)

        train_dataset = torch.utils.data.Subset(self.dataset, train_idx)
        val_dataset = torch.utils.data.Subset(self.dataset, val_idx)

        print(f"Train WSIs: {len(train_dataset)}, Val WSIs: {len(val_dataset)}")

        train_dataloader = DataLoader(
            train_dataset, batch_size=1, shuffle=True, num_workers=self.num_workers
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=1, shuffle=False, num_workers=self.num_workers
        )

        # fold毎のディレクトリ
        fold_dir = self.fold_manager.get_fold_dir(fold_idx)

        # チェックポイントコールバック
        checkpoint_callback = ModelCheckpoint(
            dirpath=fold_dir / "checkpoints",
            filename="best",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
            enable_version_counter=False
        )

        # ロガー
        logger = CSVLogger(
            save_dir=str(fold_dir),
            name="logs"
        )

        trainer = L.Trainer(
            max_epochs=50,
            accelerator='auto',
            devices=1,
            log_every_n_steps=1,
            logger=logger,
            callbacks=[checkpoint_callback]
        )

        trainer.fit(
            model=model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader
        )

        return {
            "fold_idx": fold_idx,
            "best_checkpoint": checkpoint_callback.best_model_path,
            "best_val_loss": checkpoint_callback.best_model_score
        }
