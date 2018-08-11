package my.scalaml.utils

import breeze.linalg.{DenseVector, DenseMatrix}
import scala.util.Random

object preprocessing {

  def splitTrainTestData(X: DenseMatrix[Double], y: DenseVector[Int], testRatio: Double = 0.3) = {
    val (trainIndices, testIndices) = splitTrainTestIndices(y.toArray, testRatio)
    val Xtrain = DenseMatrix(trainIndices.map{i => X(i, ::).t}: _*)
    val Xtest = DenseMatrix(testIndices.map{i => X(i, ::).t}: _*)
    val ytrain = DenseVector(trainIndices.map{i => y(i)})
    val ytest = DenseVector(testIndices.map{i => y(i)})
    (Xtrain, Xtest, ytrain, ytest)
  }

  def splitTrainTestIndices(targets: Array[Int], testRatio: Double = 0.3) = {

    val r = new Random()

    val splitWay = (targets.zipWithIndex).groupBy(_._1)
      .map{ case (c, classesAndIndices) =>
        val indices = classesAndIndices.map{_._2}
        var trainIndices = Array[Int]()
        var testIndices = Array[Int]()
        for (i <- indices)
          if (r.nextDouble() > testRatio)
            trainIndices = trainIndices :+ i
          else
            testIndices = testIndices :+ i
        (c, (trainIndices, testIndices))
    }

    val (trainIndices, testIndices) = splitWay.map{case (c, indices) => indices}
      .map{case indices => (indices._1, indices._2)}
      .foldLeft((Array[Int](), Array[Int]())){
        (finalIndices, currentOnes) =>
          (finalIndices._1 ++ currentOnes._1, finalIndices._2 ++ currentOnes._2)
      }

    (trainIndices, testIndices)

    // usage:
    //   trainIndices.map{i => targets(i)}
    //   testIndices.map{i => targets(i)}

  } // splitTrainTestIndices

} // preprocessing
