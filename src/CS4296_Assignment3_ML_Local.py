from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import col
import time
import sys

if len(sys.argv) < 3:
    print("Error: Please provide train and test paths")
    print("Usage: spark-submit CS4296_Assignment3_ML.py <train_path> <test_path> [output_path]")
    sys.exit(1)

train_path = sys.argv[1]
test_path = sys.argv[2]
output_path = sys.argv[3] if len(sys.argv) > 3 else None

# Check if running on AWS EMR (has EMR-specific environment variables)
is_aws = 'EMR_STEP_ID' in os.environ or 'AWS_EXECUTION_ENV' in os.environ

builder = SparkSession.builder.appName("CS4296_Assignment3")

if is_aws:
    # AWS EMR: use more memory
    builder = builder \
        .config("spark.executor.memory", "8g") \
        .config("spark.executor.cores", "2") \
        .config("spark.driver.memory", "4g") \
        .config("spark.memory.fraction", "0.8")
else:
    # Local PC: use less memory or defaults
    builder = builder \
        .config("spark.executor.memory", "2g") \
        .config("spark.driver.memory", "2g")

spark = builder.config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
    .getOrCreate()

print(f"Running on: {'AWS EMR' if is_aws else 'Local PC'}")

# Load the data
print("Loading training data...")
train_raw = spark.read.option("header", "true").option("inferSchema", "true").csv(train_path)
print("Loading test data...")
test_raw = spark.read.option("header", "true").option("inferSchema", "true").csv(test_path)

print("No. of Training samples:", train_raw.count())
print("No. of Test samples:", test_raw.count())

# Parse the data (pixel 0 to pixel 783)
pixel_cols = []
for i in range(784):
    pixel_cols.append(f"pixel{i}")

# Merge multiple columns into one vector
assembler = VectorAssembler(inputCols=pixel_cols, outputCol="features_raw")

# Transform both train and test data
train_data_raw = assembler.transform(train_raw).select(col("label").cast("double"), "features_raw")
test_data_raw = assembler.transform(test_raw).select(col("label").cast("double"), "features_raw")

# Scale features (mean=0, std=1)
scaler = StandardScaler(inputCol="features_raw", outputCol="features_scale", withStd=True, withMean=True)
scaler_model = scaler.fit(train_data_raw)

# For Logistic Regression and random Forest Classifier
train_data_scale = scaler_model.transform(train_data_raw).select("label", col("features_scale").alias("features"))
test_data_scale = scaler_model.transform(test_data_raw).select("label", col("features_scale").alias("features"))

# For Naive Bayes
train_data_nb = train_data_raw.select(col("label"), col("features_raw").alias("features"))
test_data_nb = test_data_raw.select(col("label"), col("features_raw").alias("features"))

print("Data preparation complete")

def evaluate_model(model, model_name, train_set, test_set):
    
    print("\n" + "="*60)
    print("Training: ", model_name)
    
    start = time.time()
    trained = model.fit(train_set)  
    elapsed = time.time() - start
    
    # Predict on test set
    test_predict = trained.transform(test_set)
    
    # Calculate test accuracy
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    test_acc = evaluator.evaluate(test_predict)
    
    # Predict and calculate training accuracy
    train_predict = trained.transform(train_set)
    train_acc = evaluator.evaluate(train_predict)
    
    print("Training time: %.2f seconds" % elapsed)
    print("Training accuracy: %.4f" % train_acc)
    print("Test accuracy: %.4f" % test_acc)
    
    return {
        "name": model_name,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "time": elapsed
    }

print("\n" + "="*60)
print("ALGORITHM 1 - LOGISTIC REGRESSION")
print("="*60)
# Set family to "multinomial" since there are mutiple classes (digits 0-9)
lr = LogisticRegression(maxIter=50, regParam=0.01, family="multinomial")
results = []
results.append(evaluate_model(lr, "Logistic Regression", train_data_scale, test_data_scale))

print("\n" + "="*60)
print("PARAMETER IMPACT ON ALGORITHM 1 - LOGISTIC REGRESSION")
print("="*60)
print("\n---Testing different maxIter values---")
for maxIter in [10, 30, 50, 100]:
    lr_test = LogisticRegression(maxIter=maxIter, regParam=0.01, family="multinomial")
    results.append(evaluate_model(lr_test, "Logistic Regression (maxIter=%d)" % maxIter, train_data_scale, test_data_scale))

print("\n---Testing different regParam values---")
for reg in [0.0, 0.001, 0.01, 0.1, 1.0]:
    lr_test = LogisticRegression(maxIter=50, regParam=reg, family="multinomial")
    results.append(evaluate_model(lr_test, "Logistic Regression (regParam=%.3f)" % reg, train_data_scale, test_data_scale))

print("\n" + "="*60)
print("ALGORITHM 2: RANDOM FOREST CLASSIFIER")
print("="*60)
# Seed to ensure reproducibility
rf = RandomForestClassifier(numTrees=50, maxDepth=10, seed=42)
results.append(evaluate_model(rf, "Random Forest", train_data_scale, test_data_scale))

print("\n" + "="*60)
print("ALGORITHM 3: NAIVE BAYES CLASSIFIER")
print("="*60)
nb = NaiveBayes(smoothing=1.0, modelType="multinomial")
results.append(evaluate_model(nb, "Naive Bayes", train_data_nb, test_data_nb))

# Result Comparision
print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)
print("\n%-30s | %-12s | %-12s | %-10s" % ("Model", "Train Acc", "Test Acc", "Time(s)"))
print("-" * 70)

for r in results:
    print("%-30s | %-12.4f | %-12.4f | %-10.2f" % (r["name"], r["train_acc"], r["test_acc"], r["time"]))

# Save results if output path provided
if output_path:
    with open(output_path, "w") as f:
        f.write("\n" + "="*60 + "\n")
        f.write("RESULTS SUMMARY\n")
        f.write("="*60 + "\n")
        f.write(f"{'Model':<35} {'Train Acc':<12} {'Test Acc':<12} {'Time(s)':<10}\n")
        f.write("-"*75 + "\n")
        for r in results:
            f.write(f"{r['name']:<35} {r['train_acc']:<12.4f} {r['test_acc']:<12.4f} {r['time']:<10.2f}\n")
    print(f"\nResults saved to {output_path}")

spark.stop()