package my.scalaml.NaiveBayes

import breeze.linalg.{DenseVector, DenseMatrix, sum}
import breeze.linalg.sum


import breeze.numerics.log

trait BaseDiscreteNaiveBayes extends BaseNaiveBayes {

  var alpha: Double = 0.0
  var classCounts: DenseVector[Double] = DenseVector[Double]()
  var featureCounts: DenseMatrix[Double] = new DenseMatrix(0, 0, Array.empty[Double])
  var classLogProb: DenseVector[Double] = DenseVector[Double]()
  var featureLogProb: DenseMatrix[Double] = new DenseMatrix(0, 0, Array.empty[Double])

  def counts(X: DenseMatrix[Double], Y: DenseMatrix[Double])
  def updateFeatureLogProb(alphaValue: Double)

  def updateClassLogProb() = {
    classLogProb = log(classCounts) - log(sum(classCounts))
  }

  def encodeTarget(y: DenseVector[Int]) = {
    classes = y.toArray.distinct.sorted
    nClasses = classes.length
    val nExample = y.size
    val Y = DenseMatrix.zeros[Double](nExample, nClasses)
    (0 until nExample).map{i => Y(i, classes(y(i))) = 1.0}
    Y
  }

  def fit(X: DenseMatrix[Double], y: DenseVector[Int]) = {
    val Y = encodeTarget(y)
    counts(X, Y)
    updateClassLogProb()
    updateFeatureLogProb(alpha)
  }

}
