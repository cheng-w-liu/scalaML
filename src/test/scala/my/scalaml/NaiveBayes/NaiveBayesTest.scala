package my.scalaml.NaiveBayes

import my.scalaml.utils.{textProcessing, preprocessing}

import breeze.linalg.sum
import scala.io.Source
import org.scalatest.FunSuite

class NaiveBayesTest extends FunSuite {

  test("Test Multinomial Naive Bayes Performance") {
    implicit def bool2int(b: Boolean) = if (b) 1 else 0

    val rawTargets = loadTarget()
    val rawFeatures = loadFeature()
    val stopWords = loadStopWords()

    val cleanRawFeatures = cleanText(rawFeatures, stopWords)

    val (target2idx, idx2target) = textProcessing.fitTarget(rawTargets)
    val (features2idx, idx2features) = textProcessing.fitFeatures(cleanRawFeatures, 10000)

    val y = textProcessing.encodeTarget(rawTargets, target2idx)
    val X = textProcessing.encodeFeatures(cleanRawFeatures, features2idx)

    val XY = preprocessing.splitTrainTestData(X, y)
    val Xtrain = XY._1
    val Xtest = XY._2
    val ytrain = XY._3
    val ytest = XY._4

    val nbModel = new MultinomialNaiveBayes(1.0)
    nbModel.fit(Xtrain, ytrain)

    val predProbTrain = nbModel.predictProba(Xtrain)
    val predTrain = nbModel.predict(Xtrain)
    val accuracyTrain = sum((predTrain :== ytrain).map { x => x.toInt }).toDouble / predTrain.length
    assert (accuracyTrain >= 0.9)

    val predProbTest = nbModel.predictProba(Xtest)
    val predTest = nbModel.predict(Xtest)
    val accuracyTest = sum((predTest :== ytest).map { x => x.toInt }).toDouble / predTest.length
    assert (accuracyTest >= 0.8)

  }

  def loadTarget() = {
    val rawTargets = Source.fromResource("target.csv").getLines.toList
    rawTargets
  }

  def loadFeature() = {
    val rawFeatures = Source.fromResource("raw_features.csv").getLines.toList
    rawFeatures
  }

  def loadStopWords() = {
    val stopWords = Source.fromResource("stop_words.csv").getLines.toList
    stopWords
  }

  def cleanText(rawText: List[String], stopWords: List[String]) =
    rawText.map{ s =>
      s.split("\\s+")
        .filter(x => !stopWords.contains(x.toLowerCase))
        .mkString(" ")
        .replaceAll("[^\\p{L}\\p{Nd}]+", " ")
        .split("\\s+")
        .toList
        .map(x => x.toLowerCase)
        .filter(x => !stopWords.contains(x))
    }

}
