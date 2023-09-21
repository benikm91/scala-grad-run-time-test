package scalagrad

import breeze.linalg.DenseMatrix
import scalagrad.api.forward.ForwardMode
import scalagrad.api.matrixalgebra.MatrixAlgebraDSL
import scalagrad.api.reverse.ReverseMode
import scalagrad.auto.breeze.BreezeDoubleMatrixAlgebraDSL

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Paths}

object ScalaGradAsymptoticPerformanceTest:

  import scalagrad.TimeUtil.*

  @main def scalaGradAsymptoticPerformanceTestMain(experimentId: Int, n: Int, nRows: Int): Unit =

    println(s"RUN EXPERIMENT with id=$experimentId iterations=$n and nRows=$nRows")

    val experimentName: "com2" = "com2"

    /**
     * Create an experiment where we $n times apply the identity dot product onto a matrix.
     *
     * @param n Number of iterations
     */
    def createComplexExperiment(n: Int)(alg: MatrixAlgebraDSL)(s1: alg.Matrix, s2: alg.Matrix): alg.Scalar =
      var (t1, t2) = (s1, s2)
      for (_ <- 0 until n) {
        t1 = t1 *:* t2
        t2 = t1 + t2
      }
      t1.sum + t2.sum

    def createExperiment(n: Int)(alg: MatrixAlgebraDSL)(s1: alg.Matrix, s2: alg.Matrix): alg.Scalar =
      experimentName match
        case "com2" => createComplexExperiment(n)(alg)(s1, s2)

    // val mode = ForwardMode
    val mode = ReverseMode

    // WARM UP
    {
      println(f"WARM UP for $experimentName ...")
      for (i <- 1 to 1) {
        val m1 = DenseMatrix.zeros[Double](nRows, nRows)
        val m2 = DenseMatrix.zeros[Double](nRows, nRows)
        val f = createExperiment(n)
        f(BreezeDoubleMatrixAlgebraDSL)(m1, m2)
        val df = mode.derive(f)
        df(BreezeDoubleMatrixAlgebraDSL)(m1, m2)
      }
      println(f"...WARM UP DONE")
    }

    val f = createExperiment(n)
    // We take zero as inputs, so intermediates stay zero and we are sure that we don't get NaNs
    val m1 = DenseMatrix.zeros[Double](nRows, nRows)
    val m2 = DenseMatrix.zeros[Double](nRows, nRows)
    val (_, elapsedTimeForwardPass) = timeMeasure {
      f(BreezeDoubleMatrixAlgebraDSL)(m1, m2)
    }
    System.gc(); Thread.sleep(200)  // GC
    val (_, elapsedTimeDerive) = timeMeasure {
      val df = mode.derive(f)
      df(BreezeDoubleMatrixAlgebraDSL)(m1, m2)
    }
    println(f"$n \t $elapsedTimeForwardPass \t $elapsedTimeDerive \t ${elapsedTimeForwardPass / elapsedTimeDerive}")

    val nElements = nRows * nRows
    def sampleHeader: String = "experiment_id,iterations,elapsed_time_forward_pass,elapsed_time_derive,n_elements"
    val csv = sampleHeader + "\n" + f"$experimentId,$n,$elapsedTimeForwardPass,$elapsedTimeDerive,$nElements"
    Files.write(Paths.get(f"out/scalagrad_run_${experimentName}_${experimentId}_${n}_${nElements}_performance_test.csv"), csv.getBytes(StandardCharsets.UTF_8))
