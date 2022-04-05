---
author: "Genrikh Ananiev"
date: "26-05-2021"
---

# Tidy stemming for classification models  in master data management (MDM) context in R

**Problem description**

Under business conditions, narrowly specialized tasks often come across, which require a special approach because they do not fit into the standard data processing flow and constructing models.
One of these tasks is the classification of new products in master data management process (MDM).

**Example 1**

You are working in a large company (supplier) engaged in the production and / or sales of products, including through wholesale intermediaries (distributors).
Often your distributors have an obligation (in front of the company in which you work) regularly providing reporting on your own sales of your products - the so-called Sale Out.
Not always, distributors are able to report on the sold products in your company codes, more often are their own codes and their own product names that differ from the names in your system.
Accordingly, in your database there is a need to keep the table matching distributors with product codes of your account.
The more distributors, the more variations of the name of the same product. If you have a large assortment portfolio, it becomes a problem that is solved by manual labor-intensive support for such matching tables when new product name variations in your accounting system are received.

If it refers to the names of such products as to the texts of documents, and the codes of your accounting system (to which these variations are tied) to consider classes, then we obtain a task of a multiple classification of texts. Such a matching table (which operators are maintained manually) can be considered a training sample, and if it is built on it such a classification model - it would be able to reduce the complexity of the work of operators to classify the flow of new names of existing products.
However, the classic approach to working with the text "as is" will not save you, it will be said just below.

**Example 2**

In the database of your company, data on sales (or prices) of products from external analytical (marketing) agencies are coming or from the parsing of  third-party sites. The same product from each data source will also contain variations in writing. As part of this example, the task can be even more difficult than in Example 1 because often your company's business users have the need to analyze not only your products, but also the range of your direct competitors and, accordingly, the number of classes (reference products) to which variations are tied - sharply increases.

**What is the specificity of such a class of tasks?**

First, there are a lot of classes (in fact, how many products you have so many classes) And if in this process you have to work not only with the company's products, but also competitors, the growth of such new classes can occur every day - therefore it becomes meaningless to teach one time Model to be repeatedly used to predict new products.

Secondly, the number of documents (different variations of the same product) in the classes are not very balanced: there may be one by one to class, and maybe more.

**Why does the classic approach of the multiple classification of texts work poorly?**

Consider the shortcomings of the classic text processing approach by steps:

* Stop words.

In such tasks there are no stop-words in the generally accepted concepts of any text processing package.

* Tochenization

In classic packages from the box, the division of text on words is based on the presence of punctuation or spaces.
As part of this class task (where the length of the text field input is often limited), it is not uncommon to receive product names without spaces where words are not clearly separated, but visually on the register of numbers or other language.
How to pass toochenization from the box on your favorite programming language for the name of wine "Dom.CHRISTIANmoreau0,75LPeМЂr.EtFilChablis" ?
(Unfortunately it's not a joke)

* [Stemming.](https://en.wikipedia.org/wiki/Stemming)

Product names are not text in a classic understanding of the task (such as news from sites, services reviews or newspaper headers) which is amenable to release suffix that can be discarded.
In the names of products, abbreviations are often present and the reductions of words of which are not clear how to allocate this suffix.
And there are also the names of the brands from another language group (for example, the inclusion of the brands of French or Italian) that are not amenable to a normal stemming.

* Reducing matrices.

Often, when building "Document-Term" matrices, the package of your language offers to reduce the sparsity of matrices to remove words (columns of the matrix) with a frequency below some minimum threshold.
And in classical tasks, it really helps improve quality and reduce overhead in the training of the model.
But not in such tasks. Above, I have already written that the distribution of classes is not strongly balanced - it can easily be on the same product name to the class (for example, a rare and expensive brand that has sold it for the first time and while there is only one time in the training sample). The classic approach of sparsity reduction we bring the quality of the classifier.

* Training the model.

Usually, some kind of model is trained on texts (LibSVM, naive Bayesian classifier, neural networks, or something else) which is then used repeatedly.
In this case, new classes can appear daily and the number of documents in the class can be counted as a single instance.
Therefore, it makes no sense to learn one large model for a long time using- any algorithm with online training, for example, a KNN classifier with one nearest neighbor, is enough.


Next, we will try to compare the classification of the traditional approach with the classification based on the proposed package. 
We will use tidytext as an auxiliary package.

# Case example

```{r,message=FALSE}
devtools::install_github(repo = 'https://github.com/edvardoss/abbrevTexts')
library(abbrevTexts)
library(tidytext) # text proccessing
library(dplyr) # data processing
library(stringr) # data processing
library(SnowballC) # traditional stemming approach
library(tm) #need only for tidytext internal purpose
```

The package includes 2 data sets on the names of wines: the original names of wines from external data sources - "rawProducts" and the unified names of wines written in the standards for maintaining the company's master data - "standardProducts".
The rawProducts table has many spelling variations of the same product, these variations are reduced to one product in standardProducts through a many-to-one relationship on the "standartId" key column.
PS Variations in the "rawProducts" table are generated programmatically, but with the maximum possible similarity to how product names come from external various sources in my experience (although somewhere I may have overdone it)

```{r}
data(rawProducts, package = 'abbrevTexts')
head(rawProducts)
```

![01_tab](https://user-images.githubusercontent.com/16530092/127452591-739c2717-42eb-417b-9111-1cd8793279b3.png)

```{r}
data(standardProducts, package = 'abbrevTexts')
head(standardProducts)
```

![02_tab](https://user-images.githubusercontent.com/16530092/127453633-29f7453d-c8ad-47a0-b8cf-3b3ca7f18b47.png)

## Train and  test split

```{r}
set.seed(1234)
trainSample <- sample(x = seq(nrow(rawProducts)),size = .9*nrow(rawProducts))
testSample <- setdiff(seq(nrow(rawProducts)),trainSample)
testSample
```

## Create dataframes for 'no stemming mode' and 'traditional stemming mode'

```{r}
df <- rawProducts %>% mutate(prodId=row_number(), 
                             rawName=str_replace_all(rawName,pattern = '\\.','. ')) %>% 
  unnest_tokens(output = word,input = rawName) %>% count(StandartId,prodId,word)

df.noStem <- df %>% bind_tf_idf(term = word,document = prodId,n = n)

df.SnowballStem <- df %>% mutate(wordStm=SnowballC::wordStem(word)) %>% 
  bind_tf_idf(term = wordStm,document = prodId,n = n)
```

## Create document terms matrix

```{r}
dtm.noStem <- df.noStem %>% 
  cast_dtm(document = prodId,term = word,value = tf_idf) %>% data.matrix()

dtm.SnowballStem <- df.SnowballStem %>% 
  cast_dtm(document = prodId,term = wordStm,value = tf_idf) %>% data.matrix()
```

## Create knn model for 'no stemming mode' and calculate accuracy

```{r}
knn.noStem <- class::knn1(train = dtm.noStem[trainSample,],
                          test = dtm.noStem[testSample,],
                          cl = rawProducts$StandartId[trainSample])
mean(knn.noStem==rawProducts$StandartId[testSample])
```

**Accuracy is: 0.4761905 (47%)**

## Create knn model for 'stemming mode' and calculate accuracy

```{r}
knn.SnowballStem <- class::knn1(train = dtm.SnowballStem[trainSample,],
                               test = dtm.SnowballStem[testSample,],
                               cl = rawProducts$StandartId[trainSample])
mean(knn.SnowballStem==rawProducts$StandartId[testSample])
```

**Accuracy is: 0.5 (50%)**

# abbrevTexts primer

Below is an example on the same data but using the  functions from abbrevTexts package


## Separating words by case

```{r}
df <- rawProducts %>% mutate(prodId=row_number(), 
                             rawNameSplitted= makeSeparatedWords(rawName)) %>% 
        unnest_tokens(output = word,input = rawNameSplitted)
print(df)
```
![03_tab](https://user-images.githubusercontent.com/16530092/127463009-51feb1b7-2677-410d-b334-905d64cfdadf.png)

As you can see, the tokenization of the text was carried out correctly: not only transitions from upper and lower case when writing together are taken into account, but also punctuation marks between words written together without spaces are taken into account.

## Creating a stemming dictionary based on a training sample of words

After a long search among different stemming implementations, I came to the conclusion that traditional methods based on the rules of the language are not suitable for such specific tasks, so I had to look for my own approach. As a result, I came to the most optimal solution, which was reduced to unsupervised learning, which is not sensitive to the text language or the degree of reduction of the available words in the training sample.

The function takes a vector of words as input, the minimum word length for the training sample and the minimum fraction for considering the child word as an abbreviation of the parent word and then does the following:

1. Discarding words with a length less than the set threshold
2. Discarding words consisting of numbers
3. Sort the words in descending order of their length
4. For each word in the list:
+ 4.1 Filter out words that are less than the length of the current word and greater than or equal to the length of the current word multiplied by the minimum fraction
+ 4.2 Select from the list of filtered words those that are the beginning of the current word


Let's say that we fix min.share = 0.7
At this intermediate stage (4.2), we get a parent-child table where such examples can be found:


![04_tab](https://user-images.githubusercontent.com/16530092/127465087-2924784e-de17-4eb2-a48b-6eb68ae43c52.PNG)

Note that each line meets the condition that the length of the child's word is not shorter than 70% of the length of the parent's word.

However, there may be found pairs that can not be considered as abbreviations of words because in them different parents are reduced to one child, for example: 

![05_tab](https://user-images.githubusercontent.com/16530092/127465425-0674f796-12e1-41e8-97cd-4b2beb54cca4.PNG)

My function for such cases leaves only one pair.

Let's go back to the example with unambiguous abbreviations of words

![04_tab](https://user-images.githubusercontent.com/16530092/127465087-2924784e-de17-4eb2-a48b-6eb68ae43c52.PNG)


But if you look a little more closely, we see that there is a common word 'bodeg' for these 2 pairs and this word allows you to connect these pairs into one chain of abbreviations without violating our initial conditions on the length of a word to consider it an abbreviation of another word: 

bodegas->bodeg->bode

So we come to a table of the form:

![06_tab](https://user-images.githubusercontent.com/16530092/127467414-75967142-dc17-483a-b51f-f328b6ee6132.PNG)

Such chains can be of arbitrary length and it is possible to assemble from the found pairs into such chains recursively. Thus we come to the 5th stage of determining the final child for each participant of the constructed chain of abbreviations of words

5. Recursively iterating through the found pairs to determine the final (terminal) child for all members of chains
6. Return the abbreviation dictionary 

The makeAbbrStemDict function is automatically paralleled by several threads loading all the processor cores, so it is advisable to take this point into account for large volumes of texts.

```{r}
abrDict <- makeAbbrStemDict(term.vec = df$word,min.len = 3,min.share = .6)
head(abrDict) # We can see parent word, intermediate results and total result (terminal child)
```

![07_tab](https://user-images.githubusercontent.com/16530092/127615663-846228b0-e1d8-4da0-9f36-8a66e022333b.PNG)


The output of the stemming dictionary in the form of a table is also convenient because it is possible to selectively and in a simple way in the "dplyr" paradigm to delete some of the stemming lines.

Lets say that we wont to exclude parent word "abruzz" and terminal child group "absolu" from stemming dictionary:

```{r}
abrDict.reduced <- abrDict %>% filter(parent!='abruzz',terminal.child!='absolu')
print(abrDict.reduced)
```

![10_tab](https://user-images.githubusercontent.com/16530092/127614613-e4e8659e-9168-44c4-a85f-79be74654c18.PNG)

Compare the simplicity and clarity of this solution with what is offered in stackoverflow:

[Text-mining with the tm-package - word stemming](https://stackoverflow.com/questions/16069406/text-mining-with-the-tm-package-word-stemming)

## Stem using abbreviate dictionary 
```{r}
df.AbbrStem <- df %>% left_join(abrDict %>% select(parent,terminal.child),by = c('word'='parent')) %>% 
    mutate(wordAbbrStem=coalesce(terminal.child,word)) %>% select(-terminal.child)
print(df.AbbrStem)
```

![08_tab](https://user-images.githubusercontent.com/16530092/127467969-6350e09a-69ec-433c-a933-fe8e526cce65.PNG)

## TF-IDF for stemmed words

```{r}
df.AbbrStem <- df.AbbrStem %>% count(StandartId,prodId,wordAbbrStem) %>% 
  bind_tf_idf(term = wordAbbrStem,document = prodId,n = n)
print(df.AbbrStem)
```

![09_tab](https://user-images.githubusercontent.com/16530092/127615790-e477b417-f8f2-4321-be3d-90406b4ee48d.PNG)


## Create document terms matrix

```{r}
dtm.AbbrStem <- df.AbbrStem %>% 
  cast_dtm(document = prodId,term = wordAbbrStem,value = tf_idf) %>% data.matrix()
```

## Create knn model for 'abbrevTexts mode' and calculate accuracy

```{r}
knn.AbbrStem <- class::knn1(train = dtm.AbbrStem[trainSample,],
                                test = dtm.AbbrStem[testSample,],
                                cl = rawProducts$StandartId[trainSample])
mean(knn.AbbrStem==rawProducts$StandartId[testSample]) 
```

**Accuracy for "abbrevTexts": 0.8333333 (83%)**

As you can see , we have received significant improvements in the quality of classification in the test sample.
Tidytext is a convenient package for a small courpus of texts, but in the case of a large courpus of texts, the "AbbrevTexts" package is also perfectly suitable for preprocessing and normalization and usually gives better accuracy in such specific tasks compared to the traditional approach.



