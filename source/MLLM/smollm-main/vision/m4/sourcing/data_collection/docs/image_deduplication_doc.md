# Image deduplication


# Methods

The main information of an image is contained in its low frequencies, the high frequencies providing detail. The main methods to perform image deduplication in a fast way, and whose quality is not so bad, are based on this observation.

## Average hashing (AHash)

- Convert the image to grayscale.
- Reduce the size of the image. To have a hash of 64, shrink the image to 8x8. It will remove the high frequencies.
- Compute the average value of the 64 pixels.
- For each pixel, replace its value by 1 if it is larger than the average or by 0 otherwise.
- Unroll the image to obtain the hash.

A hamming distance is used to compare two hashes. It is fast, but not efficient against minor modifications of the image. Lot of false positives.

## Perceptual hashing (PHash)

- Convert the image to grayscale.
- Reduce the size of the image, to 32x32 for example. This step is done to simplify the DCT computation and not because it is needed to reduce the high frequencies.
- Compute the 32x32 DCT, and keep the top left 8x8, which represents the low frequencies.
- Compute the average value of the top left 8x8, and exclude the DC coefficient.
- For each of the 64 pixels, replace its value by 1 if it is larger than the average or by 0 otherwise.
- Unroll the image to obtain the hash.

It is slower than AHash, but more robust to minor modifications of the image. Less false positives than AHash in practice.

## Difference hashing (DHash)

- Convert the image to grayscale.
- Reduce the size to 9x8, essentially to remove the high frequencies.
- For each pixel, compare its value to the one at the right. Replace by 1 if it is larger or by 0 otherwise, until ending with a 8x8 image.
- Unroll the image to obtain the hash.
- Optional: repeat the steps by comparing pixels on the columns instead of on the rows, and concatenate the new hash with the previous one to obtain a 128-bit hash, and in practice reduce the the number of false positives.

It is as fast as AHash, with less false positives. Less accuracy than PHash.

## Wavelet hashing (WHash)
Same as PHash, but uses DWT instead of DCT. It is way faster than PHash, a bit less than AHash and DHash, but produces way more false positives.


# Libraries

## [`imagededup`](https://github.com/idealo/imagededup)

It duplicates with the algorithms: CNN, Perceptual hashing, Difference hashing, Wavelet hashing, Average hashing.

## [`imagehash`](https://github.com/JohannesBuchner/imagehash)

It supports: Perceptual hashing, Difference hashing, Wavelet hashing, Average hashing, HSV color hashing (colorhash), Crop-resistant hashing.

## [`image-match`](https://github.com/ProvenanceLabs/image-match)

It implements the Image signature algorithm.

## [`imgdupes`](https://github.com/knjcode/imgdupes)

It supports Perceptual hashing (using only the 8x8 DCT low-frequency values including the first term), Difference hashing, Wavelet hashing, Average hashing, Perceptual hashing org (using only the 8x8 DCT low-frequency values and excluding the first term since the DC coefficient can be significantly different from the other values and will throw off the average). It uses `imagehash` except for Perceptual hashing org.

## [`simhash-py`](https://github.com/seomoz/simhash-py)

Implements the SimHash algorithm, as well as a solution for the Hamming distance problem.

## [`faiss`](https://github.com/facebookresearch/faiss)

Efficient similarity search and clustering of dense vectors.

## [`hnswlib`](https://github.com/nmslib/hnswlib)

Fast approximate nearest neighbors


# Research papers

## Interesting papers

- [Duplicate Discovery on 2 Billion Internet Images (2013)](https://people.csail.mit.edu/celiu/pdfs/CVPR13-bigvis-dupdiscovery.pdf):
This paper presents a method to provide a hash to an image, by considering different scales and splitting the image, computing the average pixel values of each block, gathering everything, performing a PCA (trained on a portion of the total of the images), and quantizing it by putting to 0 or 1 the coefficients of the PCA depending on if they are below or above the average PCA coefficient values.<br>
Moreover, this paper performs an $\epsilon$-clustering (complexity $\mathcal{O}(n^2)$!) to find clusters by comparing the distances of the PCA signatures (before quantization), and then another loop to improve the results by merging clusters where representatives have PCA signatures (after quantization) with a low Hamming distance.<br>
I am really surprised that the complexity $\mathcal{O}(n^2)$ worked for them on 2B images, even if they considered 24-bit hashes.<br>
They also found 1/4 of the images being duplicated (icons, ads, ...).

- [D2LV: A Data-Driven and Local-Verification Approach for Image Copy Detection (2021)](https://arxiv.org/pdf/2111.07090.pdf):
The authors are the winners of the Image Similarity Challenge proposed by Facebook. They used neural networks with an unsupervised pretraining, followed by a training with both a triplet and a cross entropy loss. They used a combination of global and local features (it is often the case in other approaches).

- [3rd Place: A Global and Local Dual Retrieval Solution to Facebook AI Image Similarity Challenge (2021)](https://arxiv.org/pdf/2112.02373.pdf):
This paper presents a method for checking if two images are similar. It uses a combination of global and local features. The global features are obtained with a Transformer pre-trained on ImageNet and trained with a triplet loss, and the local features are from the SIFT algorithm.

- [Detecting Near-Duplicates for Web Crawling (2007)](https://www2007.org/papers/paper215.pdf):
This paper verifies that the SimHash algorithm (for text) creates hashes such that near duplicates have close Hamming distances. Also, and this is what is most important for us, it gives an efficient solution for the Hamming distance problem to efficiently determine closest simhashes (i.e not in quadratic time) via random projections or permutations + interpolation search.

- [An image signature for any kind of image (2002)](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.104.2585&rep=rep1&type=pdf):
This presents the Image signature algorithm. We start by cropping the image to essentially remove the constants areas around. Then, we divide the image into a grid of patches. For each patch, we compute the average gray level, and compare it to the neighboring patches. We replace the patch by 8 values, all between -2, -1, 0, 1 and 2, depending on the difference between the level of the patch and its 8 neighbors. We then concatenate all the outputs to obtain a 648-bit array. The difference between two hashes is computed with the euclidean distance.<br>
This algorithm is interesting and seems fast. I like the idea of working with patches. However, the vectors are typically longer (size 648, 10x the usual size of 64).

- [A robust image hashing based on discrete wavelet transform (2017)](https://sci-hub.hkvisa.net/10.1109/icsipa.2017.8120651):
This paper introduces a new hashing method, based on image normalization, DWT and SVD. Some ideas can be taken to improve the simple PHash algorithm.


## Less interesting papers

- [Fast and accurate near-duplicate image elimination for visual sensor networks (2016)](https://journals.sagepub.com/doi/pdf/10.1177/1550147717694172):
Interesting paper, but I find it really similar to [Duplicate Discovery on 2 Billion Internet Images (2013)](https://people.csail.mit.edu/celiu/pdfs/CVPR13-bigvis-dupdiscovery.pdf) in the sense that they are also doing a two-step method with global and then local features. The clustering and nearest neighbors search is also similar. They changed the hash function, and added a PageRank algorithm to find the most relevant image to keep from a cluster once it is formed, but I don't think it really matters.<br>
They provide good metrics for the evaluation.

- [Benchmarking unsupervised near-duplicate image detection (2019)](https://arxiv.org/pdf/1907.02821.pdf):
This paper makes a benchmark of existing methods for image deduplication. It is interesting to understand how to perform an evaluation.

- [Large Scale Image Deduplication (2011)](http://vision.stanford.edu/teaching/cs231a_autumn1213_internal/project/final/writeup/nondistributable/Wen_Paper.pdf):
It is based on PCA to compute the image hash. The PCA is done on a sufficiently large image collection but I am not sure performing a PCA is better than PHash.

- [Secure image deduplication through image compression (2015)](https://sci-hub.hkvisa.net/10.1016/j.jisa.2015.11.003):
It is based on a wavelet-based image compression algorithm called SPIHT. It creates the signature of an image by identifying the significant regions on the image. Interesting idea but I am not sure how this can be better than PHash.

- [Image Deduplication Based on Hashing and Clustering in Cloud Storage (2021)](https://koreascience.kr/article/JAKO202120941694290.pdf):
It presents a hashing function based on DCT (that I don't think it's better than DCT) and does the clustering with K-means, but I don't like this clustering strategy as it is really challenging to find $k$ for images.

- [CE-Dedup: Cost-Effective Convolutional Neural Nets Training based on Image Deduplication (2021)](https://arxiv.org/pdf/2109.00899.pdf):
This paper uses techniques like PHash, DHash, AHash or WHash to first deduplicate images in a training set, and then train a neural network on this to see that performances can be really close when training on the full dataset while reducing the size of the dataset by a large amount. However, it does not use a neural network to do the deduplication.

- [Efficient Cropping-Resistant Robust Image Hashing (2014)](https://sci-hub.hkvisa.net/10.1109/ares.2014.85):
This presents the Crop-resistant hashing algorithm.<br>
It is an old-school method and I’m not convinced being more robust against cropping doesn’t hurt the overall performances.

- [A lightweight virtual machine image deduplication backup approach in cloud environment (2014)](https://sci-hub.hkvisa.net/10.1109/compsac.2014.73):
It is based on the K-means algorithm but I don’t like the approach since we don’t know how to choose $k$ and we have a constraint to fit in the RAM.

- [Clustering-based acceleration for virtual machine image deduplication in the cloud environment (2016)](http://lcs.ios.ac.cn/~zhangzy/paper/JSS2016Xu.pdf):
It is essentially the same paper (also same authors) as A lightweight virtual machine image deduplication backup approach in cloud environment.

- [A duplicate image deduplication approach via Haar wavelet technology (2012)](https://sci-hub.hkvisa.net/10.1109/ccis.2012.6664249):
It is based on the wavelet decomposition instead of the DCT, which seems to perform worse.

- [A High-precision Duplicate Image Deduplication Approach (2013)](http://www.jcomputers.us/vol8/jcp0811-06.pdf):
It is based on the wavelet decomposition instead of the DCT, which seems to perform worse.

# Blog posts

https://content-blockchain.org/research/testing-different-image-hash-functions/

https://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html

https://www.hackerfactor.com/blog/index.php?/archives/529-Kind-of-Like-That.html

https://fullstackml.com/wavelet-image-hash-in-python-3504fdd282b5

https://towardsdatascience.com/understanding-locality-sensitive-hashing-49f6d1f6134

https://santhoshhari.github.io/Locality-Sensitive-Hashing/

https://en.wikipedia.org/wiki/Locality-sensitive_hashing

https://mesuvash.github.io/blog/2019/Hashing-for-similarity/

https://keras.io/examples/vision/near_dup_search/

https://drivendata.co/blog/image-similarity-winners

https://www.linkedin.com/pulse/detection-duplicate-images-using-deep-learning-aditya-sharma/

https://medium.com/mlearning-ai/a-scalable-solution-to-detect-duplicate-images-97d431c2726d

https://keras.io/examples/vision/siamese_network/
