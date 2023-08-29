ThisBuild / version := "0.0.1"
ThisBuild / scalaVersion := "3.3.0"
ThisBuild / organization := "ch.benikm91"

lazy val spireDependency = Seq(
  libraryDependencies += "org.typelevel" %% "spire" % "0.18.0"
)

lazy val breezeDependency = Seq(
  libraryDependencies ++= Seq(
    "org.scalanlp" %% "breeze" % "2.1.0",
  )
)

lazy val scalaGradDependency = Seq(
    libraryDependencies ++= Seq(
      ("ch.benikm91"  %%  "scala-grad" % "0.9.0"),
      ("ch.benikm91"  %%  "scala-grad-api" %"0.9.0"),
      ("ch.benikm91"  %%   "scala-grad-auto-breeze" % "0.9.0"),
    )
)

lazy val basicSettings = Seq(
    Compile / scalaSource := baseDirectory.value / "src",
    Compile / resourceDirectory := baseDirectory.value / "res",
    Test / scalaSource := baseDirectory.value / "test",
    Test / parallelExecution := true,
)

lazy val root = (project in file("."))
  .settings(
    name := "scala-grad-impl",
    basicSettings,
    spireDependency,
    breezeDependency,
    scalaGradDependency,
  )
