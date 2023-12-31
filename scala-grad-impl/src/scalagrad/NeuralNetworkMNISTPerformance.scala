package scalagrad

import breeze.linalg.*
import scalagrad.Util.*
import scalagrad.api.matrixalgebra.MatrixAlgebraDSL
import scalagrad.auto.breeze.BreezeFloatMatrixAlgebraDSL
import scalagrad.auto.predef.PredefFloatMatrixAlgebraDSL

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Paths}
import scala.annotation.tailrec

object NeuralNetworkMNISTPerformance:

    /**
     * MNIST showcase with performance measurements to compare to other automatic differentiation libraries.
     * Notes for correct comparision:
     * - We are using float point precision
     * - MNIST data is preloded into memory
     */

    import TimeUtil.*

    @main def neuralNetworkReverseModePerformance() =

        import NeuralNetworkMNIST.{Parameters, crossEntropy, d, neuralNetwork}

        // CONFIG
        val batchSize = 64
        val nHiddenUnitList = List(512) // List(8, 16, 32, 64, 128, 256)
        val nRuns = 1
        val epochs = 10
        val lr = 0.01f

        val nFeatures = MNISTDataSet.nFeatures
        val nOutputUnits = MNISTDataSet.nLabels

        val (xsTrainDouble, ysTrainDouble) = preprocess(MNISTDataSet.loadTrain, batchSize)
        val xsTrain = xsTrainDouble.map(_.map(_.toFloat))
        val ysTrain = ysTrainDouble.map(_.map(_.toFloat))
        val eagerData = xsTrain.zip(ysTrain).toList

        def loss(xs: DenseMatrix[Float], ys: DenseMatrix[Float])(alg: MatrixAlgebraDSL)(
            p: Parameters[alg.ColumnVector, alg.Matrix]
        ): alg.Scalar =
            given alg.type = alg
            val ysHat = neuralNetwork(using alg)(alg.lift(xs), p)
            crossEntropy(using alg)(alg.lift(ys), ysHat)

        @tailrec
        def miniBatchGradientDescent
        (data: List[(DenseMatrix[Float], DenseMatrix[Float])])
        (
            p: Parameters[DenseVector[Float], DenseMatrix[Float]],
            lr: Float, // learning rate
        ): Parameters[DenseVector[Float], DenseMatrix[Float]] =
            if data.isEmpty then p
            else
                // get next batch
                val (xsBatch, ysBatch) = data.head

                // derive the loss function for the current batch
                val dLoss = d(loss(xsBatch, ysBatch))(BreezeFloatMatrixAlgebraDSL)
                val dP = dLoss(p)
                
                miniBatchGradientDescent(data.tail)(
                    // apply gradient descent update on weights
                    Parameters(
                        p.firstW0 - lr * dP.firstW0,
                        p.firstWs - lr * dP.firstWs,
                        p.lastW0 - lr * dP.lastW0,
                        p.lastWs - lr * dP.lastWs, 
                    ),
                    lr,
                )

        def getRandomParametersFloat(nFeatures: Int, nHiddenUnits: Int, nOutputUnits: Int):
            Parameters[DenseVector[Float], DenseMatrix[Float]] =
            val rand = scala.util.Random()
            Parameters(
                DenseVector.fill(nHiddenUnits)(rand.nextFloat() - 0.5f),
                DenseMatrix.fill(nFeatures, nHiddenUnits)(rand.nextFloat() - 0.5f),
                DenseVector.fill(nOutputUnits)(rand.nextFloat() - 0.5f),
                DenseMatrix.fill(nHiddenUnits, nOutputUnits)(rand.nextFloat() - 0.5f),
            )

        def accuracy(yHatProp: DenseMatrix[Float], yM: DenseMatrix[Float]): Float =
            val yHat = yHatProp(*, ::).map(x => argmax(x))
            val y = yM(*, ::).map(x => argmax(x))
            val correct = yHat.toArray.zip(y.toArray).map((yHat, y) => if yHat == y then 1 else 0).sum
            correct.toFloat / yHat.length

        // Print current performance of Parameters on test data
        def logPerformance(xsTest: Seq[DenseMatrix[Float]], ysTest: Seq[DenseMatrix[Float]])(p: Parameters[DenseVector[Float], DenseMatrix[Float]]): Float =
            given BreezeFloatMatrixAlgebraDSL.type = BreezeFloatMatrixAlgebraDSL
            val ysHatTest = xsTest.map(xs => neuralNetwork(xs, p))
            val accuracyTestBatch = ysHatTest.zip(ysTest).map((ysHat, ys) => accuracy(ysHat, ys)).toList
            val accuracyTest = accuracyTestBatch.sum / accuracyTestBatch.length
            val lossTestBatch = ysHatTest.zip(ysTest).map((ysHat, ys) => crossEntropy(ys, ysHat)).toList
            val lossTest = lossTestBatch.sum / lossTestBatch.length
            println(
                List(
                    f"testLoss=${lossTest}%.1f",
                    f"testAcc=${accuracyTest * 100}%3f",
                ).mkString("\t")
            )
            accuracyTest * 100

        val (xsTestDouble, ysTestDouble) = preprocess(MNISTDataSet.loadTest, 32)
        val xsTest = xsTestDouble.map(_.map(_.toFloat))
        val ysTest = ysTestDouble.map(_.map(_.toFloat))

        case class Sample(experimentId: Int, epoch: Int, elapsedTrainTime: Double, elapsedTestTime: Double, testAccuracy: Float)

        // WARM UP JVM
        (1 to 2).foldLeft(getRandomParametersFloat(nFeatures, 8, nOutputUnits)) {
            case (currentParams, epoch) =>
                println(f"WARUM UP epoch ${epoch}")
                val nextParams = miniBatchGradientDescent(eagerData)(currentParams, lr)
                val testAccuracy = logPerformance(xsTest, ysTest)(nextParams)
                nextParams
        }
        Thread.sleep(2000) // sleep 2s between runs => GC

        var measurements = List[Sample]()

        nHiddenUnitList.map(nHiddenUnits => {
            var measurements = List[Sample]()

            // Run miniBatchGradientDescent {epochs} number of times
            (1 to nRuns).foreach(experimentId => {
                // Initialize weights
                val initialParams = getRandomParametersFloat(nFeatures, nHiddenUnits, nOutputUnits)
                (1 to epochs).foldLeft(initialParams) {
                    case (currentParams, epoch) =>
                        println(f"epoch ${epoch}")
                        val (nextParams, elapsedTrainTime) = timeMeasure {
                            miniBatchGradientDescent(eagerData)(currentParams, lr)
                        }
                        val (testAccuracy, elapsedTestTime) = timeMeasure {
                            logPerformance(xsTest, ysTest)(nextParams)
                        }
                        measurements = measurements.appended(Sample(
                            experimentId, epoch, elapsedTrainTime, elapsedTestTime, testAccuracy
                        ))
                        nextParams
                }
                Thread.sleep(2000)  // sleep 2s between runs => GC
            })

            def sampleHeader: String = "experiment_id,epoch,elapsed_train_time,elapsed_test_time,test_acc"
            def sampleToCSV(s: Sample): String = f"${s.experimentId},${s.epoch},${s.elapsedTrainTime},${s.elapsedTestTime},${s.testAccuracy}"
            val csv = sampleHeader + "\n" + measurements.map(sampleToCSV).mkString("\n")
            Files.write(Paths.get(f"out/scalagrad_results_$nHiddenUnits.csv"), csv.getBytes(StandardCharsets.UTF_8))
        })


object TimeUtil:

    import java.util.concurrent.TimeUnit

    def time[R](block: => R): R =
        val t0 = System.nanoTime()
        val result = block    // call-by-name
        val t1 = System.nanoTime()
        val ds = TimeUnit.SECONDS.convert(t1 - t0, TimeUnit.NANOSECONDS)
        println("Elapsed time: " + ds + "s")
        result

    def timeMeasure[R](block: => R): (R, Double) =
        val t0 = System.nanoTime()
        val result = block    // call-by-name
        val t1 = System.nanoTime()
        val ds = (t1 - t0) / 1_000_000_000.0
        (result, ds)
