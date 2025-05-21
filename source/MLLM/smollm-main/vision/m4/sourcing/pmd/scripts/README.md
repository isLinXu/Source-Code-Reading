# Creating image PMD - points of entry

Some subsets are handled on JZ, and some others are handled on thomas-m4-pmd GCP VM.

To launch the ones on JZ (Conceptual captions and Google WIT), from this folder, launch the slurm job:
```batch
sbatch jz_image_pmd.slurm --mail-type=ALL --mail-user=victor@huggingface.co # The two last arguments are optional
```
The job will expect you to have the `m4` repo under `$WORK/code`, and will save data under `$cnw_ALL_CCFRSCRATCH/general_pmd/image`. You can then upload things to the bucket to save them:
```bash
gsutil -m rsync -r $cnw_ALL_CCFRSCRATCH/general_pmd/image/ gs://science-m4/general_pmd/image/
```

To launch the ones on `thomas-m4-pmd`, from this folder, run the following comamnds:
```bash
mkdir -p $HOME/general_pmd/image
# Optionally, activate your favorite conda env
python pmd.py
gsutil -m rsync -r $HOME/general_pmd/image/ gs://science-m4/general_pmd/image/
```

Once the creation are done, you can sanity check how many images are missing using the script `check_none_ims.py`.

A lot of the subsets require manually downloading files and putting them in the right folder. Note that they are files that are automatically downloaded from `facebook/pmd` (https://huggingface.co/datasets/facebook/pmd), please make sure you have filled the authorization wall so that the download can automatically happen.

|Subset|File location|Where to put and what to do|
|--|--|--|
|LocalizedNarrativesFlickr30K|http://shannon.cs.illinois.edu/DenotationGraph/data/index.html|Download "Flickr 30k images" and decompress the tar.gz into `~/.cache/m4/flickr30k`|

## Tarring the `downloaded_images` folders in `~/.cache/m4/`

```bash
find . -type d -maxdepth 1 -mindepth 1 -exec basename \{} ./ \; | parallel --verbose -j 16 --progress "tar -zcf {1}.tar.gz {1}"
```

## Helper scripts

If you want to know how many images were downloaded in a subfolder:
```bash
find {dataset_name} -type f -regextype egrep -regex "{dataset_name}/downloaded_images/[0-9a-f]{3}/[0-9a-f]{3}/[0-9a-f]{64}" | wc -l
```

If you want to remove all .lock files from a subfolder:
```bash
find {dataset_name} -type f -regextype egrep -regex "{dataset_name}/downloaded_images/[0-9a-f]{3}/[0-9a-f]{3}/[0-9a-f]{64}\.lock" | xargs -I {} rm {}
```

If you want to remove all tmp_files from a subfolder:
```bash
find {dataset_name} -type f -regextype egrep -regex "{dataset_name}/downloaded_images/temp-.*" | xargs -I {} rm {}
```

If you want to tar and split the subfolder (typically before pushing to a bucket):
```bash
tar -cvf - {dataset_name} | split -b 1G -d -a 7 --additional-suffix=.tar - "{dataset_name}_part-"
```

Note: on MAC, `split` has to be replaced with `gsplit` (pleaase install it via `brew install coreutils`)
