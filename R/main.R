#' makeAbbrStemDict
#'
#' Creating a dictionary of words to transform  to their most short overall reduction (multi-core based on a basic parallel package)
#' @param term.vec  Character vector of input words
#' @param min.len The  length of word to stop of abbreviate searching
#' @param min.share The minimal ratio of lengths of two words to consider the first word as an abbreviation of the second word
#' @return data frame with 3 columns: input words, intermediate stemming words, output  words
#' @examples
#' words <- c('Stemming','Stemmin','Stemm','Stem','Stemmi','SomeWord','St')
#' dict <- makeAbbrStemDict(term.vec=words,min.share=.8)
#' View(dict)
#' @export
makeAbbrStemDict<-function(term.vec, min.len=3, min.share=NA){
  vls <- unique(term.vec)
  vls <- vls[nchar(vls)>=min.len]
  vls <- vls[grepl(pattern="^\\D.+",x=vls)]
  vls <- vls[order(nchar(vls),decreasing = T)]
  len.vls <- nchar(vls)
  min.share <- ifelse(is.na(min.share),0,min.share)
  cls <- parallel::makeCluster(parallel::detectCores()-1)
  parallel::clusterExport(cl = cls,varlist = c("vls","len.vls","min.share"),envir=environment())
  fn <- function(i){
     vl.subs <- vls[len.vls>=min.share*nchar(i)&len.vls<nchar(i)]
    if(length(vl.subs)>0){
      vl.flag <- pmatch(x = vl.subs,table = i,duplicates.ok = F)
      vl.res <- vl.subs[!is.na(vl.flag)]
      if(length(vl.res)==0){
        vl.res <- NA
      }
    } else {
      vl.res <- NA
    }
    return(vl.res)
  }
  child <- parallel::parSapply(cl = cls,X = vls,FUN = fn,USE.NAMES = F)
  res <- data.frame(parent=vls,child,stringsAsFactors = F)
  res <- res[!duplicated(res$child)&!is.na(res$child),]
  parallel::clusterExport(cl = cls,varlist = "res",envir = environment())
  fn_terminal.leaf <- function(vl.prnt){
    vl.chld <- res$child[res$parent==vl.prnt]
    if(length(vl.chld)==0){
      return(vl.prnt)
    } else {
      fn_terminal.leaf(vl.chld)
    }
  }
  res$terminal.child <- parallel::parSapply(cl = cls,X = res$parent,FUN = fn_terminal.leaf)
  parallel::stopCluster(cls);rm(cls)
  res <- res[order(res$terminal.child),]
  return(res)
}


#' makeSeparatedWords
#'
#' Separating merged words using letter case, language and digits
#' @param txt  Character vector of input texts
#' @return Character vector of separated words
#' @examples
#' products <- data.frame(life_case=c('"camel style"','forgotten CAPS LOCK','some words in upper','english~not english','digits and words'),
#'                        primer=c('ChateauBaronBellevueCoteDeBourgBordeaux', 'rIOJAcRIANZAbODEGASmARQUES', 'PINOTnoirBOURGOGNEdomaineMASSE', 'colombard ugniбланles betesрозовоеcomte Tolosan', 'Valencisco Rioja Blanco, Bodeguera de Valencisco0.75vol'))
#' products$splitted <- makeSeparatedWords(products$primer)
#' View(products)
#' @export
makeSeparatedWords <- function(txt){
  txt<-gsub(pattern = "([[:lower:]]{2,})([[:upper:]]{1})",replacement = "\\1 \\2",x=txt,ignore.case = F)
  txt<-gsub(pattern = "\\.",replacement = ". ",x=txt,ignore.case = F)
  txt<-gsub(pattern = "([[:upper:]]{1}[[:lower:]]{1,})([[:upper:]]{1}[[:lower:]]{1,})",replacement = "\\1 \\2",x=txt,ignore.case = F)
  txt<-gsub(pattern = "([[:upper:]]{2,})([[:lower:]]{1,})",replacement = "\\1 \\2",x=txt,ignore.case = F)
  txt<-gsub(pattern = "([[:upper:]]{2,})([[:lower:]][[:upper:]]+)",replacement = "\\1 \\2",x=txt,ignore.case = F,perl = T) # Perl важен
  txt<-gsub(pattern = "([a-z]{2,})([^a-z]{2,})",replacement = "\\1 \\2",x=txt,ignore.case = T,fixed=F)
  txt<-gsub(pattern = "([^a-z]{2,})([a-z]{2,})",replacement = "\\1 \\2",x=txt,ignore.case = T,fixed=F)
  txt <- gsub(pattern = "([[:digit:]])([[:alpha:]])",replacement = "\\1 \\2",x = txt,ignore.case = F,fixed = F)
  txt <- gsub(pattern = "([[:alpha:]])([[:digit:]])",replacement = "\\1 \\2",x = txt,ignore.case = F,fixed = F)
  return(txt)
}


#' makeReducedSparceMatrix
#'
#' Removal from an unnecessary column matrix (words) that exist in the training set and do not exist in the prediction set.
#' After that (optional): removal of empty rows that were formed after removing columns in the previous step (such rows do not carry predictive strength but occupy a place in memory)
#' @param dfmSparceMatrix  Document-term (feature) matrix. May be sparse or traditional.
#' @param trainIndex True/False vector is the same length as the number of rows of the initial matrix. Truth - is record from training sample.
#' @param removePotentiallyEmptyRows A sign of whether it is necessary to cut the matrix on the empty rows (if they appeared after the column removal)
#' @return Function return list with reduced matrix and reduced (if removePotentiallyEmptyRows==T) trainIndex for model prediction
#' @export

makeReducedSparceMatrix <- function(dfmSparceMatrix,trainIndex, removePotentiallyEmptyRows=T){
  if(!is.logical(trainIndex)){
    stop('trainIndex must be True/False vector')
  }
  if(length(trainIndex)!=nrow(dfmSparceMatrix)){
    stop('length of trainIndex must be same as matrix row number')
  }
  terms.train <- colSums(x = dfmSparceMatrix[trainIndex,],na.rm = T)>0
  terms.predict <-colSums(x = dfmSparceMatrix[!trainIndex,],na.rm = T)>0
  terms.intersect <- terms.train&terms.predict
  rows.TrainPredict.termsIntersect <- rowSums(x=dfmSparceMatrix[,terms.intersect],na.rm = T)>0
  rows.TrainIndexReduced <- trainIndex&rows.TrainPredict.termsIntersect
  if(removePotentiallyEmptyRows){
    return(list(reducedMatrix=dfmSparceMatrix[rows.TrainIndexReduced,terms.intersect],
                reducedTrainIndex=rows.TrainIndexReduced))
  } else {
    return(list(reducedMatrix=dfmSparceMatrix[,terms.intersect],
                reducedTrainIndex=NA))
  }
}
