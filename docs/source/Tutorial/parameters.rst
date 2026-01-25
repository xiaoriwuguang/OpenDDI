Parameters
===========

This section provides a comprehensive list of all command-line parameters available for the OpenDDI main script. All parameters and their default values are directly extracted from :doc:`parms_setting.py <../API/parms_setting>`.

General Parameters
-------------------

.. code-block:: bash

    --no-cuda                              # Force use CPU (default=False)
    --device <device>                      # Device selection: 'auto', 'cuda', or 'cpu' (default='cuda')
    --workers <int>                        # Number of workers (default=4)

Training Parameters
-------------------

.. code-block:: bash

    --lr <float>                          # Learning rate (default=1e-3)
    --dropout <float>                     # Dropout rate (default=0.3)
    --weight_decay <float>                # Weight decay for optimizer (default=5e-4)
    --batch <int>                         # Batch size (default=32768)
    --epochs <int>                        # Number of training epochs (default=150)

Model Parameters
----------------

.. code-block:: bash

    --model <model_name>                   # Select model architecture (default='DSNDDI')

**Available Models (20 total):**

.. code-block:: text

    MRCGNN, GOGNN, ZeroDDI, DDIMDL, TIGER, ConvLSTM, MVA,
    MUFFIN, DeepDDI, DDKG, SumGNN, LaGAT, KGNN, PHGLDDI,
    MMDGDTI, DSNDDI, ExDDI, MIRACLE, CASTER, MKGFENN

.. code-block:: bash

    --network_ratio <float>                # Network ratio parameter (default=0.1)
    --loss_ratio1 <float>                 # Loss ratio 1 (default=1.0)
    --loss_ratio2 <float>                 # Loss ratio 2 (default=0.05)
    --loss_ratio3 <float>                 # Loss ratio 3 (default=0.1)
    --hidden1 <int>                       # Hidden layer 1 dimension (default=512)
    --hidden2 <int>                       # Hidden layer 2 dimension (default=256)

Dataset Parameters
------------------

.. code-block:: bash

    --matrix <matrix_name>                 # Choose the interaction matrix dataset (default='Ryus')

**Available Datasets (9 total):**

.. code-block:: text

    binary, zhangddi, ChCh-Miner, multi, zeroddi, 
    Dengs, Ryus, multilabel, twosides


.. code-block:: bash

    --modality <modality_name> [<modality_name> ...]  # Modality types (can specify multiple)

**Available Modalities (6 total):**

.. code-block:: text

    smiles, sequence, 3d, mechanism, text, drkg

**Default:** smiles, sequence, 3d, mechanism, text

**Note:** Use ``--modality smiles sequence 3d`` to specify multiple modalities (space-separated).

.. code-block:: bash

    --features <int> [<int> ...]          # Feature dimensions for each modality (default=[300, 320, 512, 320, 768])
    --dimensions <int>                     # Total feature dimension (default=512)
    --num_classes <int>                    # Number of classes (default=-1, auto-detect)
    --matrix_dir <path>                    # Directory path for interaction matrices (default='openddi/../datasets/matrix/')
    --embedding_dir <path>                 # Directory path for embeddings (default='openddi/../datasets/emb/')
    --origin <bool>                        # Whether to use original model implementation (default=False)
    --general <bool>                       # Whether to perform generalization experiments (default=False)

Modal Splits Parameters
------------------------

.. code-block:: bash

    --modal_splits <str>                   # Modal dimension splits, comma-separated (default=None)
                                          # Example: "1024,768,256,128"

Noise Parameters
-----------------

.. code-block:: bash

    --noise_std <float>                    # Standard deviation for Gaussian noise on input features (default=0.0)
    --noise_ratio <float>                  # Ratio of noisy labels in training set (default=0.0)

Sparsity Parameters
--------------------

.. code-block:: bash

    --sparse_drop_rate <float>             # Feature random zeroing ratio (default=0.0)
    --sparse_sample_rate <float>          # Training set label sampling ratio (default=0.0)

Zero-Shot Learning Parameters
------------------------------

.. code-block:: bash

    --event_sem_path <path>                # Path to event semantic embeddings (.npy/.csv), default is one-hot
    --zs_protocol <protocol>               # Zero-shot protocol (choices: none, CZSL, GZSL; default='none')
    --zs_ratio <float>                     # Zero-shot ratio (default=0.3)
    --zs_seed <int>                        # Random seed for zero-shot experiments (default=1)

Alignment Loss Parameters
--------------------------

.. code-block:: bash

    --lambda_align <float>                 # Alignment loss weight (default=1.0)
    --lambda_u_pair <float>               # Unpaired loss weight (default=0.1)
    --lambda_u_event <float>              # Unpaired event loss weight (default=0.1)
    --uniform_t <float>                   # Uniform temperature parameter (default=2.0)

Task Parameters
----------------

.. code-block:: bash

    --task <task_type>                     # Task type (currently only 'train_xxxx' available, default='train_xxxx')

Embedding File Mappings
------------------------

Based on the code, each modality maps to specific embedding files:

.. code-block:: text

    smiles    → smiles_embeddings.pt
    sequence  → sequence_embeddings.pt
    3d        → 3d_embeddings.pt
    mechanism → mechanism_embeddings.pt
    text      → text_embeddings.pt
    drkg      → drkg_embeddings.pt
