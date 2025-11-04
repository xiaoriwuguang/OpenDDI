## 🚀 Quick Start

Follow these steps to get started with OpenDDI:

### **Step 1: Clone the Repository**

```
git clone <REPO-URL>
cd OpenDDI
```

### **Step 2: Install Dependencies**

#### General Dependencies

You can install the general dependencies:

```
conda env create -f openddi.yml
pip install torch-scatter==2.0.7 torch-sparse==0.6.9 -f https://data.pyg.org/whl/torch-1.7.0+cu110.html
```

### **Step 3: Run the Main Script**

After installing the dependencies, you can run the main script using the following command:

```
python openddi/main.py --model <model_name> --matrix <matrix_name> --modality<modality_name> --epochs <epochs> --batch <batch>
```

#### Optional arguments:

- `--model <model_name>`: Model choice, e.g., `MKGFENN`, `CASTER`, `MMDGDTI`, etc.
- `--matrix <matrix_name>`: Interaction matrix, e.g., `binary`, `ChCh-Miner`, `Ryus`, etc.
- `--modality<modality_name>`: Interaction matrix, e.g., `binary`, `ChCh-Miner`, `Ryus`, etc.
- `--epochs <epochs>`: Number of epochs to run.
- `--batch <batch>`: The batch size to use.

Note that the above list includes only a subset of the available parameters. For more parameters and their descriptions, please refer to the `openddi/parms_setting.py` file.

### Example Command:

To run the **MKGFENN** model on the **Ryus** DDI dataset with **smiles sequence 3d mechanism text** modalities, you can run the following command:

```
python openddi/main.py --model MRCGNN --matrix Ryus --modality smiles sequence 3d mechanism text --epochs 100 --batch 4096
```

This command will:

- Use the **MKGFENN** model
- Use the **Ryus** dataset
- Use the **smiles sequence 3d mechanism text** modalities
- Train for **100 epochs** with a **batch size of 4096**