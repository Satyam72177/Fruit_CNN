from pyspark.sql import SparkSession
from pyspark.sql.functions import col, split, explode, trim

# Create Spark session
spark = SparkSession.builder.appName("ConditionalSplitAllCases").getOrCreate()

# Sample data
data = [
    ("101", "1st/2nd", "A/B"),     # Case 1: both split
    ("102", "Single", "X/Y"),      # Case 2: category split only
    ("103", "2nd/3rd", "Single"),  # Case 3: record_type split only
    ("104", "OnlyOne", "Single")   # Case 4: no split
]
df = spark.createDataFrame(data, ["id", "record_type", "category"])

df.show(truncate=False)
# +---+-----------+--------+
# |id |record_type|category|
# +---+-----------+--------+
# |101|1st/2nd    |A/B     |
# |102|Single     |X/Y     |
# |103|2nd/3rd    |Single  |
# |104|OnlyOne    |Single  |
# +---+-----------+--------+

# Case 1: Both fields contain "/"
df_both = (
    df.filter(col("record_type").contains("/") & col("category").contains("/"))
      .withColumn("record_type", explode(split(col("record_type"), "/")))
      .withColumn("category", explode(split(col("category"), "/")))
      .withColumn("record_type", trim(col("record_type")))
      .withColumn("category", trim(col("category")))
)

# Case 2: Only category has "/"
df_cat_only = (
    df.filter(~col("record_type").contains("/") & col("category").contains("/"))
      .withColumn("category", explode(split(col("category"), "/")))
      .withColumn("record_type", trim(col("record_type")))
      .withColumn("category", trim(col("category")))
)

# Case 3: Only record_type has "/"
df_rec_only = (
    df.filter(col("record_type").contains("/") & ~col("category").contains("/"))
      .withColumn("record_type", explode(split(col("record_type"), "/")))
      .withColumn("record_type", trim(col("record_type")))
      .withColumn("category", trim(col("category")))
)

# Case 4: Neither has "/"
df_normal = df.filter(~col("record_type").contains("/") & ~col("category").contains("/"))

# Union all
df_final = df_both.unionByName(df_cat_only).unionByName(df_rec_only).unionByName(df_normal)

df_final.show(truncate=False)
# +---+-----------+--------+
# |id |record_type|category|
# +---+-----------+--------+
# |101|1st        |A       |
# |101|2nd        |B       |
# |102|Single     |X       |
# |102|Single     |Y       |
# |103|2nd        |Single  |
# |103|3rd        |Single  |
# |104|OnlyOne    |Single  |
# +---+-----------+--------+