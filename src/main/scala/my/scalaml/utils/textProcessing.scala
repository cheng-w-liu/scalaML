package my.scalaml.utils
import breeze.linalg.{DenseVector, DenseMatrix}

object textProcessing {

  def fitTarget(rawTargets: List[String]) = {
    val targets = rawTargets.distinct.sorted
    val target2idx = targets.zipWithIndex.toMap
    val idx2target = (targets.indices zip targets).toMap
    (target2idx, idx2target)
  }

  def encodeTarget(rawTargets: List[String], target2idx: Map[String, Int]) = {
    val buff = rawTargets.map{ t => target2idx(t) }
    val y = DenseVector(buff: _*)
    y
  }

  def fitFeatures(rawFeatures: List[List[String]], vocSize: Int = 10000) = {

    val valueCounts = rawFeatures.foldLeft(scala.collection.mutable.Map[String, Int]()) {
      (counts, oneExample) =>
        oneExample.foreach{ w =>
          if (counts.contains(w)) counts(w) += 1
          else counts(w) = 1
        }
        counts
    }

    val tmp = valueCounts.toArray.sortBy(-_._2).take(vocSize)
    println("top 10 words:")
    println(tmp.slice(0, 10).toList)
    println("bottom 10 words:")
    println(tmp.slice(vocSize-10, vocSize).toList)
    val kept = valueCounts.toArray.sortBy(-_._2).take(vocSize).toMap

    val features = rawFeatures
          .map{x => x.distinct.toArray.filter(x => kept.contains(x))}
          .foldLeft(Array[String]()){(feats, trainExample) =>
            (feats ++ trainExample).distinct
          }
          .sorted
    val features2idx = features.zipWithIndex.toMap
    val idx2features = (features.indices zip features).toMap
    (features2idx, idx2features)
  }

  def encodeFeatures(rawFeatures: List[List[String]], feature2idx: Map[String, Int]) = {
    val buff = rawFeatures.map{ feats =>
      val x = DenseVector.zeros[Double](feature2idx.size)
      feats.foreach{ word =>
        if (feature2idx.contains(word))
          x(feature2idx(word)) = 1.0
      }
      x
    }
    val dataMatrix = DenseMatrix(buff.map(_.toArray):_*)
    dataMatrix
  }
  
}
