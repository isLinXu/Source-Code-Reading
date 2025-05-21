# Data handling logic - documentation

For training, data are stored in sharded web tars (i.e. webdataset format), and each of these shards live on s3. For training, we download and decode the shards into in-memory samples, pack these samples to form training examples (i.e. sequences), and yield these sequences of data through the data loader to the model training loop. Each of these three logic is handled in a different file.

+ `dataset_utils.py` -> *decode the shards into in-memory samples*
+ `packing.py` -> *pack these samples to form training examples (i.e. sequences)*
+ `dataset.py` -> *yield these sequences of data through the data loader to the model training loop*

In this md file, we give useful details about each of these components, in addition to the docstrings in each of these files.

## Decode the shards into in-memory samples

Web tar shards are downloaded with the help of the script `m4/experiments/pretraining/vloom/common/webdataset_get_file.sh`. It handles downloading the curent shards in a temp folder and temp name, yield it into a readable data stream, and delete the temp shard once we are done with it.

`dataset_utils.py` mostly handles reading this stream and decode this stream into samples. The highest level entry point is the function `get_webdataset`. It defines the series of steps (splitting to nodes and data workers, decoding to samples, and shuffling). The `shuffle*` arguments are series of arguments that control for pseudo shuffling the data. The main change between different types of dataset is the decoding part. For image/text pairs, it requires loading an image and a text field, for web documents, it requires loading an arbitrary (but ordered) sequence of images and texts, etc. Each decoding function defines the necessary utilities to decode the web tar shards that have been previously saved and uploaded to s3, and sanity check them.

The main drawback off webdataset that we have never solved is the determinism: every time we resume a training, we have no guarantees that the sample yielded have not already been seen. Essentially, we don't have control over the data order.

Note that all the functions defined in `dataset_utils.py` are easily debuggable in the vscode debugger.

## pack these samples to form training examples (i.e. sequences)

Depending on the type of data, specific samples packing method are used:
+ `split_pack_and_pad_iqa_finetuning` -> question/answer/image triplets, specific for vqa fine-tuning
+ `split_pack_and_pad_ocr` -> ocr documents that require specific pdf decoding
+ `split_pack_and_pad_pairs` -> image/caption pairs
+ `split_pack_and_pad_sft` -> chatbot formatted SFT
+ `split_pack_and_pad_webdocs` -> multimodal documents

### PMD (or any other image/text pairs dataset)

This is the `split_pack_and_pad_pairs` method.

PMD contains ~70M image-text pairs originally introduced in FLAVA paper as a combination of publicly available datasets. To use PMD we follow these steps:

- Each image is represented as an `<image>` token and added to text. We add `<fake_token_around_image>` before and after the sequence of `<image>` tokens.
- After each image-text pair, an end of document token is added.
- We continue adding the text containing `<image>...<image>` + the caption until we cross the `max_seq_len` specified by the parameters. If we cross it, we add the current pair to next sample and pad the current sample up until the `max_seq_len`. This ensures that there is no image with incomplete text.

### CM4 (or any other multimodal documents dataset)

This is the `split_pack_and_pad_webdocs` method.

In Idefics2, the sequence of two image would be represented by `<fake_token_around_image><image><image>...<image><fake_token_around_image><image><image>...<fake_token_around_image>`.

**Sampling sub-sequences** (copy-pasted from the code comments)

Following Flamingo (i.e. Idefics1) we sample a random sequence of a specific length from a document and then takes a maximum of `max_num_images` that belong to that sequence.

Computing the start index for the sub-sequence to sample is done by skewing the sampling towards sub-sequences that contain images. The main idea is to give a bonus to tokens that are closely before an image token, so that these tokens have more chance to be sampled.

Bonuses are computed for each image, which means a given token can receive bonuses from multiple images if this token is closely preceding multiple images.
We sum all the bonuses and L1 normalized along the seq_len axis to get a probability distribution.
Each token start with a regular bonus of 1, which corresponds to the uniform distribution over the sequence when there are no bonuses added.

*For the sake of simplicity, we describe the algorithm in the case where images take only ONE visual token (N in practise) in addition to the `<fake_token_around_image>` before and after.*

Now the remaining question is which precedding tokens do we distribue bonuses to.
We first observe that for the sampled sub-sequence to be considered valid (i.e. sub-sequence contains an image), the start index can only be among [image_idx - max_seq_len + 1, image_idx].
For the sake of the explanation, let's split the [image_idx - max_seq_len + 1, image_idx] interval in 3 parts: left, middle and right (in increasing order).
If we give bonuses to the tokens just before the image (right part), then we are favoring p_next=0 because only the tokens after the image have an image to attend to.
In practice, images will tend to be at the beginning of the sampled sub-sequence.
If we give bonuses very far before the image (left part), then we are favoring p_next=1 because only the tokens before the image gave an image to attend to.
In practice, images will tend to be at the end of the sampled sub-sequence.
To avoid choosing favoring p_next=0 or p_next=1, we can give bonuses to the tokens in the middle part.
In practise, images will tend to be in the middle of the sampled sequence.

Ultimately, we don't want to skew the distribution fed to model in that way (i.e. whether images are in the beginning, middle or end of the sampled sub-sequence),
and have all these cases represented equally in the data. So the easiest is to distribute a bonus to all of the max_seq_len tokens preceding the image.

### SFT datasets

This is the `split_pack_and_pad_sft` method.
It is relatively similar to `split_pack_and_pad_pairs`, the main addition is handling samples with no images.

### image/question/answer triplets datasets

This is the `split_pack_and_pad_iqa_finetuning` method.
It is relatively similar to `split_pack_and_pad_pairs`, the main addition is handling samples with two separated question/answer fields, which is relevant in the context of fine-tuning (in particular vqa fine-tunings).

### OCR datasets

This is the `split_pack_and_pad_ocr` method.
It is relatively similar to `split_pack_and_pad_pairs`, the main addition is handling the specific file decoding for ocr datasets.

## Attention masks

In Idefics2, the attention masks are fully auto-regressive, meaning that tokens are attended from left to right in an auto-regressive fashion. We try having full attention (vs left-to-right attention) on the image sequences with no significant performance improvement (at the cost of a more complicated attention mask). This attention mask is referred to as `attention_mask`, which is not to be mixed with the `image_attention_mask` which handles padding for the navit-style image resolution and image ratio preserving vision encoder.

## Yield these sequences of data through the data loader to the model training loop

Except for a few legacy tests, we do not pre-process the datasets but do that on the fly (notably packing).

For testing out processing on the fly, we need an iterable dataset as our packing strategies generally are applied on the batch level. Alternatively, we can do it in collate function of the dataloader as it usually gets a batch but we then risk again facing [this PyTorch issue](https://github.com/pytorch/pytorch/issues/13246) as the dataset will return some text strings by default.

For the iterable dataset, some of the following things are tricky. I have also added our current solutions for each of the situation.

- We would want that each process's dataloader loads **different subset of the data** otherwise there can be overlap and the processes will tend to load the same data. This can be achieved in two ways: (i) different ordering of the dataset for each process (ii) different subset of the data for each of the processes. [DistributedSampler](https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler) from PyTorch uses (ii) so we will also target that.
- **Shuffling** is tricky. Ideally, we need it to be reproducible but also different for each process and each epoch. We will rely on the rank as well as the current epoch to make it reproducible based on the rank and changing with each epoch. Once we have the indices we want to sample, we can shuffle them in a deterministic way. *Having reproducibility and determinism was possible in our FIRST implementation that relied on HF dataset, but after we switched to web dataset, it was not possible anymore. We did switch to web dataset over dataset because reading data was too much of a bottleneck and was significantly impacting throughput. It would be useful if time permits to revisit that choice now that hf datasets supports web dataset natively.*
- For uniform distribution, we would want each worker inside the dataloader to also load different subsets of data. We will rely on the local `worker_id` to the dataset to make it reproducible. For uniform fetch, we can just take indices at the gap of `num_workers` from what was returned from the previous step.

To summarize, first the indices will be divided based on the rank of the process and then further split based on the current dataloader's worker id (that's handled by `wds.SimpleShardList` and `wds.split_by_node` in `dataset_utils.py`).

Once we have a list of indices we want to sample, we can iterate over them, keep appending to a batch until we reach the batch size we want while slicing any overflows in the process to the next batch. This will ensure there is no extensive wastage. We will also drop the last uneven batch to prevent any barriers with DDP.

Note that in this case the batch size passed to the mapping (packing/padding) function can be different from the actual batch size yielded from the function. This can allow us to better utilize the mapping functions as more data in padding and packing will lead to less wastage and allow bigger operations to batched if possible.

For brevity, one alternative to the previous approach is to just take full batch length sample from the indices that we require, pack them but then only yield batch of length batch size. This will lead to some wastage.

Once we implemented the above said functionality, accelerator started to be a bottleneck in the how it handled the iterable datasets. Basically, in accelerate if you choose to not let accelerate dispatch batches for you, it [wraps the dataset](https://github.com/huggingface/accelerate/blob/469b61e0bfdb2dc3baa4df52e4e82fb6a8e48cfd/src/accelerate/data_loader.py#L216) in `IterableDatasetShard` which wastes a lot of batches but won’t probably cause any uneven batches. If you choose it to [dispatch batches for you](https://github.com/huggingface/accelerate/blob/469b61e0bfdb2dc3baa4df52e4e82fb6a8e48cfd/src/accelerate/data_loader.py#L381) then the dataloader only in the main process is used which is also a wastage (maybe can be circumvented with higher number of workers but there will be surely many zombies).

In `IterableDatasetShard`, (i) it loads the same dataloader on each of the processes, (ii) collects batches until it has the batch length of global batch size, (iii) from this global batch slice the batch corresponding to the index of the current process, (iv) dump rest of the samples. This is wastage because we processed all of that data unnecessarily just to know the right batch to sample from global batch and dump the rest of it.

Currently, since I am handling sharding myself, I don’t want either of them but I end up in uneven batches because different documents can lead to different number of batches but this doesn’t cause any wastage. One way to circumvent is to a gather and check if any worker has exhausted but this will lead us to lose a minor number of batches. We successfully implemented this strategy in our current system as an custom dataloader and it is working well for us.

Updates:

The above logic is implemented across `DataLoaderForIterableWrapperDataset` (highest-level dataloader that the training loop is iterating over), `CustomChainDataset` (handling the mixing of multiple datasets with mixture proportions that we define in config) and `IterableWrapperDataset` (iterable ofer ONE dataset type).
