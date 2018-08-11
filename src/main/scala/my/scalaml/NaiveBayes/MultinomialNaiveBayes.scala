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
