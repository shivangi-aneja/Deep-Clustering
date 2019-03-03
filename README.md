# deep-clustering
## How to run
```bash
usage: main.py [-h] [-d DATASET] [--data-dirpath DATA_DIRPATH]
               [--n-workers N_WORKERS] [--gpu GPU] [-rs RANDOM_SEED]
               [-a AUTOENCODER] [-pl PRETRAIN_LOSS][-fl FINETUNE_LOSS] [-nz LATENT_DIM] [-b BATCH_SIZE]
               [-pe PRETRAIN_EPOCHS] [-fe FINETUNE_EPOCHS] [-lr LEARNING_RATE] [-opt OPTIM] [-p PRETRAIN_MODEL_NAME]
               [-f FINETUNE_MODEL_NAME] [-alpha ALPHA] [-k K_INIT] [-mp MAX_POINTS] [-pp PREPROCESS]
               [-ppc PIXEL_PER_CELL]  [-ut UNSUPERVISED_TRAIN]

optional arguments:
  -h, --help            show this help message and exit
  -d DATASET, --dataset DATASET
                        dataset, {'mnist'} (default: mnist)
  --data-dirpath DATA_DIRPATH
                        directory for storing downloaded data (default: data/)
  --n-workers N_WORKERS
                        how many threads to use for I/O (default: 2)
  --gpu GPU             ID of the GPU to train on (or '' to train on CPU)
                        (default: 0)
  -rs RANDOM_SEED, --random-seed RANDOM_SEED
                        random seed for training (default: 1)
  -a AUTOENCODER, --autoencoder AUTOENCODER
                        autoencoder architecture name, {'mnist_autoencoder1'}
                        (default: mnist_autoencoder1)
  -pl PRETRAIN_LOSS, --pretrain_loss PRETRAIN_LOSS
                        loss function, {'mse'} (default: mse)
  -fl FINETUNE_LOSS, --finetune_loss FINETUNE_LOSS
                        loss function, {'kl_div','k_means_loss'} (default: kl_div)
  -nz LATENT_DIM, --latent-dim LATENT_DIM
                        latent space dimension for autoencoder (default: 32)
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        input batch size for training (default: 50)
  -pe PRETRAIN_EPOCHS, --pretrain_epochs PRETRAIN_EPOCHS
                        number of pretrain epochs (default: 5)
  -fe FINETUNE_EPOCHS, --finetune_epochs FINETUNE_EPOCHS
                        number of finetune epochs (default: 5)
  -lr LEARNING_RATE, --learning-rate LEARNING_RATE
                        initial learning rate (default: 0.001)
  -opt OPTIM, --optim OPTIM
                        optimizer, {'adam', 'sgd'} (default: adam)
  -p PRETRAIN_MODEL_NAME, --pretrain_model_name PRETRAIN_MODEL_NAME
                        name of pretrain model (default: 'pretain_model')
  -f FINETUNE_MODEL_NAME, --finetune_model_name FINETUNE_MODEL_NAME
                        name of finetune model (default: 'finetune_model')
  -alpha ALPHA, --alpha ALPHA
                        alpha for clustering/non-clustering loss
  -k K_INIT, --k_init  K_INIT
                        Number of restarts for K-Means Clustering
  -mp MAX_POINTS, --max_points MAX_POINTS
                        No. of points to plot during t-sne visualization
  -pp PREPROCESS, --preprocess PREPROCESS
                        Whether to pre-process the image to calcualate histogram of oriented gradients and color histogram
  -ppc PIXEL_PER_CELL, --pixel_per_cell PIXEL_PER_CELL
                        Pixel per cell to calcualate histogram of oriented gradients
  -ut UNSUPERVISED_TRAIN, --unsupervised_train UNSUPERVISED_TRAIN
                        Whether to train in an unsupervised setting or not
```

## Sample Command
```bash
python3 main.py -d mnist -a mnist_autoencoder7 -b 100 -nz 32 -pe 20 -fe 0 -p mnist_arch7_nz32_pretrain -f mnist_arch7_nz32_fine
```
## How to install
```bash
pip install -r requirements.txt
```

## How to run tests
```bash
make test
```

## Results

### MNSIT Dataset t-SNE visualisation
Gives a clustering accuracy of 70.4%
<p float="left">
  <img src="images/mnist_orig" width="420" />
  <img src="images/mnist_dl" width="420" /> 
</p>
<pre> |          Original                     |            Prediction       | </pre>


### FMNIST Dataset
Gives a clustering accuracy of 53.8%
<p float="left">
  <img src="images/fmnist_orig" width="420" />
  <img src="images/fmnist_dl" width="420" /> 
</p>
<pre> |          Original                     |            Prediction       | </pre>
