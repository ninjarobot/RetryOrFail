open System
open Microsoft.ML
open Microsoft.ML.Data

type AnsibleError () =
    [<ColumnName("Message")>]
    member val Message = Unchecked.defaultof<string> with get, set
    [<ColumnName("Job")>]
    member val Job = Unchecked.defaultof<string> with get, set
    [<ColumnName("SucceededAfterRetry")>]
    member val SucceededAfterRetry = false with get, set

type RecoverWithRetry () =
    [<ColumnName("PredictedLabel")>]
    member val WillRecover = false with get, set

[<EntryPoint>]
let main argv =
    let mlContext = new MLContext(seed=Nullable 0)
    let dataView =
        [
            AnsibleError (Message="Error connecting to gateway", Job="network-config", SucceededAfterRetry=true)
            AnsibleError (Message="Foundation error getting password", Job="foundation", SucceededAfterRetry=false)
            AnsibleError (Message="Foundation error bad password", Job="foundation", SucceededAfterRetry=false)
            AnsibleError (Message="Foundation error could not find storage account", Job="foundation", SucceededAfterRetry=false)
            AnsibleError (Message="Foundation error http timeout", Job="foundation", SucceededAfterRetry=true)
            AnsibleError (Message="Foundation error ssl peer disconnected", Job="foundation", SucceededAfterRetry=true)
            AnsibleError (Message="Unable to connect to host", Job="network-config", SucceededAfterRetry=false)
            AnsibleError (Message="Connectivity timeout", Job="network-config", SucceededAfterRetry=true)
            AnsibleError (Message="VM deployment timeout", Job="network-config", SucceededAfterRetry=false)
        ]
        |> mlContext.Data.LoadFromEnumerable
    
    let trainingPipeline =
        EstimatorChain()
            .Append(mlContext.Transforms.Conversion.ConvertType(inputColumnName="SucceededAfterRetry", outputColumnName="SucceededAfterRetryBool", outputKind=DataKind.Boolean))
            .Append(mlContext.Transforms.Conversion.MapValueToKey(inputColumnName="SucceededAfterRetryBool", outputColumnName="Label"))
            .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName="Message", outputColumnName="MessageFeaturized"))
            .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName="Job", outputColumnName="JobFeaturized"))
            .Append(mlContext.Transforms.Concatenate("Features", "MessageFeaturized", "JobFeaturized"))
            .AppendCacheCheckpoint(mlContext)
            .Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label","Features"))
            .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"))
    
    let trainedModel = trainingPipeline.Fit(dataView)
    
    let predictionEngine = mlContext.Model.CreatePredictionEngine<AnsibleError, RecoverWithRetry> (trainedModel)
    
    mlContext.Model.Save (trainedModel, dataView.Schema, "trainedModel.zip")
    
    [
        "foundation http timeout", "foundation"
        "foundation password", "foundation"
        "storage account not found", "foundation"
        "error connecting", "network-config"
        "something totally new", "foundation"
        "there was an http timeout", "foundation"
        "ssl connect timeout","network-config"
        "vm deploy","network-config"
        "http failure","software-deployment"
    ] |> List.iter (fun (msg, job) ->
            let errorToHandle = AnsibleError (Message=msg, Job=job)
            let prediction = predictionEngine.Predict(errorToHandle)
            printfn "Likely to recover from '%s': %O" msg prediction.WillRecover
        )
    0
