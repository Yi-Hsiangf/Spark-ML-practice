# Spark_ML

## Use Spark ML to analyze data in RDD, DataFrame and DataSet
In this case, I analyze bike renting data with regression tree.

In the code, you can see the Spark ML pipeline implement!
### Introduce File
* hour.csv

The data I am going to analyze!

* rdd.py 

Analyze with RDD!

* dataframe.py

Analyze with dataframe, better to look at the column of the data!

* bike_dataframe.ipynb

Analyze the data with dataframe, and easy to look at the code and result!


* DS_Bin.scala

Analyze with dataset, and use scala to implement!

* DS_Bin.zip

The whole file that is needed! The file include the DS_Bin.scala and also Build.sbt!

### Spark environment Setup
```bash
$ sudo add-apt-repository ppa:webupd8team/java
$ echo "deb https://dl.bintray.com/sbt/debian /" | sudo tee -a /etc/apt/sources.list.d/sbt.list
$ sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv 2EE0EA64E40A89B84B2DF73499E82A75642AC823
$ sudo apt-get update
$ sudo apt install oracle-java8-installer ssh python3-pip sbt
$ pip3 --no-cache-dir install numpy pandas
$ wget http://ftp.tc.edu.tw/pub/Apache/hadoop/common/hadoop-3.1.1/hadoop-3.1.1.tar.gz
$ wget http://ftp.twaren.net/Unix/Web/apache/spark/spark-2.3.2/spark-2.3.2-bin-hadoop2.7.tgz
$ ssh-keygen
$ cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
```

Edit /usr/local/hadoop/etc/hadoop/hadoop-env.sh
```bash
export JAVA_HOME = /usr/lib/jvm/java-8-oracle
```

Append following lines to ~/.bashrc
```bash
export JAVA_HOME=/usr/lib/jvm/java-8-oracle
export HADOOP_HOME=/usr/local/hadoop
export SPARK_HOME=/usr/local/spark
export PATH=$PATH:$HADOOP_HOME/bin 
export HADOOP_COMMON_LIB_NATIVE_DIR=$HADOOP_HOME/lib/native
export HADOOP_OPTS="-Djava.library.path=$HADOOP_HOME/lib"
export JAVA_LIBRARY_PATH=$HADOOP_HOME/lib/native:$JAVA_LIBRARY_PATH
export PATH=$PATH:$SPARK_HOME/bin 
export PYSPARK_PYTHON=python3
```

### To execute py file
```bash
$ spark-submit <<your py file>>
```

### To execute scala file
```bash
$Sbt package 
$spark-submit target/scala-2.11/<<your jar file>>
```
