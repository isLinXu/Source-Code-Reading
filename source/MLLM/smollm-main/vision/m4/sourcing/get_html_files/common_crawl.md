# Intersect a list of urls with the urls archived in a snapshot of Common Crawl

In this section, I want to leave a trace of the steps I followed in order to know how many oscar's english subset urls are present in the CC-MAIN-2021-25 snapshot of Common Crawl

## 1. Get the list of urls from oscar's english subset

```python
from datasets import load_dataset
import json
from tqdm import tqdm

saving_path = "/gpfswork/rech/cnw/urd43gx/urls_oscar_english/urls_oscar_english.parquet" #CHANGEME

dataset = load_dataset("oscar-corpus/OSCAR-2109", language="en", split="train", use_auth_token=True)

print("Dataset successfully loaded")

def get_urls_from_meta_column(meta_col):
   urls = [meta_item["headers"]["warc-target-uri"] for meta_item in meta_col]
   return {"url": urls}

dataset = dataset.map(get_urls_from_meta_column, batched=True, batch_size=1000, remove_columns=dataset.column_names, num_proc=25, input_columns=["meta"])

dataset.to_parquet(saving_path)
```
Note, for the following steps we need the list in a table in parquet format

## 2. Transfer the parquet table to a S3 bucket

I copied it here: `s3://m4-cc-index/urls_oscar_english/urls_oscar_english.parquet`

## 3. Create on [AWS Athena](https://aws.amazon.com/athena/) a database and a table storing the Common Crawl index

Follow the steps described here: https://commoncrawl.org/2018/03/index-to-warc-files-and-urls-in-columnar-format/

If the table `ccindex` already exist, don't forget to update it with the latest crawl snapshots by running `MSCK REPAIR TABLE ccindex`.

## 4. Create a new database and table with the oscar's english subset urls

On Athena UI, run:

```sql
CREATE DATABASE m4
```

```sql
CREATE EXTERNAL TABLE IF NOT EXISTS m4.urls_oscar_english (
         `url` string)
ROW FORMAT SERDE 'org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe'
WITH SERDEPROPERTIES (
  'serialization.format' = '1'
) LOCATION 's3://m4-cc-index/urls_oscar_english'
TBLPROPERTIES ('has_encrypted_data'='false');
```

```sql
SELECT * FROM "m4"."urls_oscar_english" limit 10;
```

|url                                                                                                                                                           |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
|https://cupe4914.com/cupe-4914-warns-that-peel-cas-needs-to-stop-putting-vital-supports-at-risk-and-get-serious-about-negotiating-a-fair-collective-agreement/|
|https://cure-ed-info.com/politics/shameful-publicity-stunt-labour-mp-hits-out-at-rishi-sunak-sit-down-with-gordon-ramsay/                                     |
|https://cure-ed-info.com/world-news/tanzanian-president-blames-lab-after-goat-papaya-test-positive-for-coronavirus/                                           |
|https://custom-essay-writer.net/2020/10/21/literature-review-hq_7h/                                                                                           |
|https://customclassicsmancaves.com/index.php?route=product/product&product_id=746                                                                             |
|https://customdistributors.com/recipe/blt-baked-potatoes/                                                                                                     |
|https://customtwit.com/30-laws-of-flow-wordpress-site-with-charlene-day/                                                                                      |
|https://customwritingshelp.com/2021/02/23/question-1-of-20-one-purpose-of-closing-entries-is-to-give-zero/                                                    |
|https://customwritingwebsite.com/2020/06/30/write-my-college-essay_kl/                                                                                        |
|https://cuttingedgechicago.com/terms-and-conditions                                                                                                           |

## 5. Join the  oscar's english subset urls table with the Common Crawl index table

```sql
CREATE TABLE "m4"."result_urls_oscar_english" WITH (
  format = 'parquet',
  external_location = 's3://m4-cc-index/result_urls_oscar_english/'
) AS
select a.*,
  b.*
from (
    select url
    from m4.urls_oscar_english
  ) as a
  left join (
    select url as url_2,
      url_host_name,
      content_mime_type,
      content_mime_detected,
      warc_filename,
      warc_record_offset,
      warc_record_length,
      warc_segment,
      crawl,
      fetch_status,
      content_languages
    from ccindex.ccindex
    where crawl = 'CC-MAIN-2021-25'
  ) as b on a.url = b.url_2
```

## 6. Get the number of oscar's english subset urls in the CC-MAIN-2021-25 snapshot
```sql
select count(*)
from m4."result_urls_oscar_english"
where url_2 is not NULL;
```
108551545

without duplicated urls:
```sql
select count(DISTINCT url)
from m4."result_urls_oscar_english"
where url_2 is not NULL;
```
106503003
