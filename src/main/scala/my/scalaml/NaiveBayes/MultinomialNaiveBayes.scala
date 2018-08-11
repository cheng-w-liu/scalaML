package my.scalaml.NaiveBayes

import breeze.linalg.{DenseVector, DenseMatrix, sum, *}
import breeze.numerics.log

class MultinomialNaiveBayes(val alphaValue: Double) extends BaseDiscreteNaiveBayes {
  alpha = alphaValue

  override def counts(X: DenseMatrix[Double], Y: DenseMatrix[Double]) = {
    featureCounts = Y.t * X
    classCounts = sum(Y(::, *)).t
  }

  override def updateFeatureLogProb(alphaValue: Double) = {
    val Ncj = featureCounts + alphaValue
    //val Nc = (0 until Ncj.rows).map{c => sum(Ncj(c, ::))}
    val Nc = DenseVector( (0 until Ncj.rows).map{c => sum(Ncj(c, ::))}: _* )
    val logNcj = log(Ncj)
    featureLogProb = logNcj(::, *) - log(Nc)
  }

  override def logLikelihood(X: DenseMatrix[Double]) = {
    val ll = X * featureLogProb.t
    ll(*, ::) + classLogProb
  }
}

/*
class MultinomialNaiveBayes extends BaseNaiveBayes {

  var classes = Array[Int]()
  var target = DenseVector[Double]()
  var features = DenseMatrix((0.0))


  override def getClasses = DenseVector(0, 1)

  override def logLikelihood(X: DenseMatrix[Double]) = {

    DenseMatrix((1.0, 2.0), (3.0, 4.0))

    //val ll = 1.0
    //println("Inside logLikelihood")
    //ll
  }

}
*/