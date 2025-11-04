import os
import argparse
import torch
import random
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def set_random_seed(seed, deterministic=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

set_random_seed(1, deterministic=True)
CODE_DIR = os.path.dirname(os.path.abspath(__file__))

def settings():
    parser = argparse.ArgumentParser()

    # ---------------- 基础训练参数 ----------------
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Force use CPU')
    parser.add_argument('--device', type=str, choices=['auto', 'cuda', 'cpu'],
                        default='cuda', help="Device: 'auto' -> cuda if available else cpu")
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--batch', type=int, default=16384)
    parser.add_argument('--epochs', type=int, default=100)

    # 输出文件目录
    default_fig_dir = os.path.join(CODE_DIR, '..', 'results')
    os.makedirs(default_fig_dir, exist_ok=True)

    # 模态切分（可选）
    parser.add_argument('--modal_splits', type=str, default=None,
                        help='各模态维度, 逗号分隔, 之和需等于 args.dimensions, 例如 "1024,768,256,128"')

    # ---------------- 模型/任务选择 ----------------
    parser.add_argument('--model', type=str,
                        choices=['MRCGNN','GOGNN','ZeroDDI','DDIMDL','TIGER','ConvLSTM','MVA',
                                 'MUFFIN','DeepDDI','DDKG','SumGNN','LaGAT','KGNN','PHGLDDI',
                                 'MMDGDTI','ExDDI','MIRACLE','CASTER','MKGFENN'],
                        default='MKGFENN')
    parser.add_argument('--network_ratio', type=float, default=0.1)
    parser.add_argument('--loss_ratio1', type=float, default=1.0)
    parser.add_argument('--loss_ratio2', type=float, default=0.05)
    parser.add_argument('--loss_ratio3', type=float, default=0.1)
    parser.add_argument('--hidden1', type=int, default=512)
    parser.add_argument('--hidden2', type=int, default=256)

    # ---------------- 数据集/模态 ----------------
    parser.add_argument('--features', type=int, nargs='+', default=[300, 320, 512, 320, 768])
    parser.add_argument('--dimensions', type=int, default=512)
    parser.add_argument('--num_classes', type=int, default=-1)
    parser.add_argument('--matrix', type=str,
                        choices=['binary','zhangddi','ChCh-Miner',
                                 'multi','zeroddi','Dengs','Ryus',
                                 'multilabel','twosides'],
                        default='Ryus')
    parser.add_argument('--modality', type=str, nargs='+',
                        choices=['smiles','sequence','3d','mechanism','text','drkg'],
                        default=['smiles','sequence','3d','mechanism','text'])

    parser.add_argument('--matrix_dir',    type=str, default=os.path.join(CODE_DIR,'..','datasets','matrix'))
    parser.add_argument('--embedding_dir', type=str, default=os.path.join(CODE_DIR,'..','datasets','emb'))
    parser.add_argument('--task', type=str, choices=['train_xxxx'], default='train_xxxx')

    # 噪声
    parser.add_argument('--noise_std', type=float, default=0.0, help='输入特征高斯噪声 σ')
    parser.add_argument('--noise_ratio', type=float, default=0.0, help='训练集标签加噪比例')

    # 稀疏性（可选）
    parser.add_argument('--sparse_drop_rate', type=float, default=0.0)      # 特征随机置零比例
    parser.add_argument('--sparse_sample_rate', type=float, default=0.0)    # 训练集标签采样比例

    # Zero-shot
    parser.add_argument('--event_sem_path', type=str, default=None, help='K×d_e 的 .npy/.csv；缺省为 one-hot')
    parser.add_argument('--zs_protocol', type=str, choices=['none','CZSL','GZSL'], default='none')
    parser.add_argument('--zs_ratio', type=float, default=0.3)
    parser.add_argument('--zs_seed', type=int, default=1)


    # 对齐损失
    parser.add_argument('--lambda_align', type=float, default=1.0)
    parser.add_argument('--lambda_u_pair', type=float, default=0.1)
    parser.add_argument('--lambda_u_event', type=float, default=0.1)
    parser.add_argument('--uniform_t', type=float, default=2.0)

    # ---------- 先解析参数 ----------
    args = parser.parse_args()

    # ---------- 设备选择规范化 ----------
    if args.no_cuda:
        args.device = 'cpu'
    elif args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 若用户强制 --device=cuda 但系统不可用，则降级
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("[WARN] --device=cuda 但 PyTorch 未检测到 CUDA，退回 CPU。")
        args.device = 'cpu'
    args.cuda = (args.device == 'cuda')

    # ---------- 路径/映射 ----------
    # 输出文件（为每个模型单独的子目录）
    model_fig_dir = os.path.join(default_fig_dir, args.model)
    os.makedirs(model_fig_dir, exist_ok=True)
    args.out_file = os.path.join(
        model_fig_dir,
        f'{args.model}_*_{args.matrix}_{"".join(args.modality)}_*noise_fea{args.noise_std}_label{args.noise_ratio}_*sparsity_fea{args.sparse_drop_rate}_noise{args.sparse_sample_rate}.txt'
    )

    args.embedding_map = {
        'smiles':    os.path.join(args.embedding_dir, 'smiles_embeddings.pt'),
        'sequence':  os.path.join(args.embedding_dir, 'sequence_embeddings.pt'),
        '3d':        os.path.join(args.embedding_dir, '3d_embeddings.pt'),
        'mechanism': os.path.join(args.embedding_dir, 'mechanism_embeddings.pt'),
        'text':      os.path.join(args.embedding_dir, 'text_embeddings.pt'),
        'drkg':      os.path.join(args.embedding_dir, 'drkg_embeddings.pt'),
    }
    args.matrix_map = {
        'binary':     os.path.join(args.matrix_dir, 'combined ddi binary.csv'),
        'ChCh-Miner': os.path.join(args.matrix_dir, 'ChCh-Miner.csv'),
        'zhangddi':   os.path.join(args.matrix_dir, 'zhangddi.csv'),
        'multi':      os.path.join(args.matrix_dir, 'combined ddi multi.csv'),
        'Dengs':      os.path.join(args.matrix_dir, 'Dengs.csv'),
        'Ryus':       os.path.join(args.matrix_dir, 'Ryus.csv'),
        'zeroddi':    os.path.join(args.matrix_dir, 'zeroddi.csv'),
        'multilabel': os.path.join(args.matrix_dir, 'combined ddi multilabel.csv'),
        'twosides':   os.path.join(args.matrix_dir, 'origin ddi multilabel.csv'),
    }

    args.embedding_path = [args.embedding_map[m] for m in args.modality]
    args.matrix_path = args.matrix_map[args.matrix]
    args.code_dir = CODE_DIR

    # 小提示：打印一下设备/可见卡，便于你检查
    if args.cuda:
        print(f"[INFO] device=cuda | CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','ALL')} "
              f"| cuda_count={torch.cuda.device_count()}")
    else:
        print("[INFO] device=cpu")

    return args
