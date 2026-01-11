# config.py
from dataclasses import dataclass

@dataclass
class ExperimentConfig:
    exp_id: int
    description: str
    train_size: float = 0.7
    freeze_layers: int = None
    learning_rate: float = 2e-5
    batch_size: int = 32
    dropout_rate: float = 0.1
    epochs: int = 20
    max_seq_len: int = 128
    use_cls_token: bool = True
    token_position: str = "cls"
    use_class_weights: bool = True
    use_scheduler: bool = True
    gradient_accumulation: bool = False

# 定义19个实验
EXPERIMENTS = [
    # 第一组：数据量对比 (4个实验)
    ExperimentConfig(exp_id=1, description="30 percent data, frozen BERT, baseline", train_size=0.3, freeze_layers=-1),
    ExperimentConfig(exp_id=2, description="50 percent data, frozen BERT", train_size=0.5, freeze_layers=-1),
    ExperimentConfig(exp_id=3, description="70 percent data, frozen BERT", train_size=0.7, freeze_layers=-1),
    ExperimentConfig(exp_id=4, description="100 percent data, frozen BERT", train_size=1.0, freeze_layers=-1),
    
    # 第二组：冻结策略对比 (3个实验)
    ExperimentConfig(exp_id=5, description="Freeze all BERT layers", freeze_layers=-1),
    ExperimentConfig(exp_id=6, description="Freeze first 8 BERT layers", freeze_layers=8),
    ExperimentConfig(exp_id=7, description="No freezing", freeze_layers=None),
    
    # 第三组：学习率对比 (3个实验)
    ExperimentConfig(exp_id=8, description="Low learning rate 1e-5", learning_rate=1e-5),
    ExperimentConfig(exp_id=9, description="Medium learning rate 2e-5", learning_rate=2e-5),
    ExperimentConfig(exp_id=10, description="High learning rate 5e-5", learning_rate=5e-5),
    
    # 第四组：Dropout对比 (2个实验)
    ExperimentConfig(exp_id=11, description="Low dropout 0.1", dropout_rate=0.1),
    ExperimentConfig(exp_id=12, description="High dropout 0.3", dropout_rate=0.3),
    
    # 第五组：Epochs对比 (2个实验)
    ExperimentConfig(exp_id=13, description="30 epochs training", epochs=30),
    ExperimentConfig(exp_id=14, description="60 epochs training", epochs=60),
    
    # 第六组：Batch Size对比 (2个实验)
    ExperimentConfig(exp_id=15, description="Small batch size 16", batch_size=16, gradient_accumulation=True),
    ExperimentConfig(exp_id=16, description="Large batch size 64", batch_size=64),
    
    # 第七组：Token策略对比 (3个实验)
    ExperimentConfig(exp_id=17, description="Mean pooling", use_cls_token=False, token_position="mean"),
    ExperimentConfig(exp_id=18, description="Max pooling", use_cls_token=False, token_position="max"),
    ExperimentConfig(exp_id=19, description="First+Last concat", use_cls_token=False, token_position="first_last"),
]