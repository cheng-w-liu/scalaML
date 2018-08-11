package my.scalaml.utils

import breeze.linalg.{DenseVector, DenseMatrix, max, sum}
import breeze.numerics.{exp, log}

object numerical {

  def logsumexp(X: DenseMatrix[Double]) = {
    val tmp = (0 until X.rows).map{ i =>
      val maxVal = max(X(i, ::))
      log(sum(exp(X(i, ::) - maxVal))) + maxVal
    }
    val Z = DenseVector(tmp: _*)
    Z
    //val logProba = X(::, *) - Z  // X(::, *) means for each column
    //logProba
  }


}
