package my.scalaml.NaiveBayes
import my.scalaml.utils.numerical._

import breeze.linalg.{DenseVector, DenseMatrix, argmax, *}
import breeze.numerics.exp

trait BaseNaiveBayes {

  var classes: Array[Int] = Array[Int]()
  var nClasses: Int = 0

  def logLikelihood(X: DenseMatrix[Double]): DenseMatrix[Double]

  def predict(X: DenseMatrix[Double]) = {
    val ll = logLikelihood(X)
    val predictedClasses = DenseVector(
      (0 until ll.rows)
        .map{ i => argmax(ll(i, ::)) }
        .map{ i => classes(i) }: _*)
    predictedClasses
  }

  def predictLogProba(X: DenseMatrix[Double]) = {
    val ll = logLikelihood(X)
    val Z = logsumexp(ll)
    val logProba = ll(::, *) - Z  // X(::, *) means for each column
    logProba
  }

  def predictProba(X: DenseMatrix[Double]) = {
    val logProba = predictLogProba(X)
    val proba = exp(logProba)
    proba
  }
}
