import Pkg
using Pkg
Pkg.add("HTTP")
Pkg.add("JSON")
Pkg.add("DataFrames")

using HTTP, JSON, DataFrames

dataset_id = 61  # Replace with the dataset ID you are interested in.

# Get the task associated with the dataset ID
url = "https://www.openml.org/api/v1/json/task/list/data_id/$(dataset_id)"
response = HTTP.get(url)
task_data = JSON.parse(String(response.body))["tasks"]["task"]
task_id = task_data[1]["task_id"]


evaluation_url = "https://www.openml.org/api/v1/json/evaluation/list/function/area_under_roc_curve/task/$task_id/sort_order/desc/limit/1"
evaluation_response = HTTP.get(evaluation_url)
evaluation_data = JSON.parse(String(evaluation_response.body))["evaluations"]["evaluation"][1]

best_run_id = evaluation_data["run_id"]


flow_id = evaluation_data["flow_id"]
flow_url = "https://www.openml.org/api/v1/json/flow/$(flow_id)"
flow_response = HTTP.get(flow_url)
flow_data = JSON.parse(String(flow_response.body))["flow"] 

# print(flow_data)

function summarize_component(component_input::Pair{String, Any})
    component_info = component_input.second

    println("       Parameters:")

    if isa(component_info["parameter"], Dict{String, Any})
        println("           Name: $(component_info["parameter"]["name"])")
        print("  Data Type: ", (component_info["parameter"]["data_type"]))
        print("  Default Value: ", (component_info["parameter"]["default_value"]))
        print("  Description: ", (component_info["parameter"]["description"]))
    elseif !isempty(component_info["parameter"])
        for parameter in component_info["parameter"]
            println("           Name: $(parameter["name"])")
            print("  Data Type: ", isempty(parameter["data_type"]) ? "Unknown" : parameter["data_type"])
            print("  Default Value: ", isempty(parameter["default_value"]) ? "None" : parameter["default_value"])
            print("  Description: ", isempty(parameter["description"]) ? "None" : parameter["description"])
            
        end
    else
        println("   No parameters.")
    end
    println()
end

function display_flow_info(flow_info)
    println("Name: $(flow_info["name"])")
    if (haskey(flow_info, "custom_name"))
        println("Custom name: $(flow_info["custom_name"])")
    end
    println()
    println("Description: $(flow_info["description"])")

    # println("Parameters:")
    # for (parameter, value) in flow_info["parameter"]
    #     println("  $(parameter): $(value)")
    # end

    println()
    println("Components:")
    for (component, value) in flow_info["component"]
        name = component.second

        println("   Component: $(name)")
        summarize_component(value)
    end

end

display_flow_info(flow_data)

# Pair{String, Any}("flow", Dict{String, Any}("language" => "English", "dependencies" => "sklearn==0.18.1\nnumpy>=1.6.1\nscipy>=0.9", "name" => "openmlstudy14.preprocessing.ConditionalImputer", "id" => "7660", "version" => "6", "description" => "Automatically created scikit-learn flow.", "parameter" => Any[Dict{String, Any}("name" => "axis", "data_type" => Any[], "default_value" => "0", "description" => Any[]), Dict{String, Any}("name" => "categorical_features", "data_type" => Any[], "default_value" => Any[], "description" => Any[]), Dict{String, Any}("name" => "copy", "data_type" => Any[], "default_value" => "true", "description" => Any[]), Dict{String, Any}("name" => "fill_empty", "data_type" => Any[], "default_value" => "0", "description" => Any[]), Dict{String, Any}("name" => "missing_values", "data_type" => Any[], "default_value" => "\"NaN\"", "description" => Any[]), Dict{String, Any}("name" => "strategy", "data_type" => Any[], "default_value" => "\"mean\"", "description" => Any[]), 
# Dict{String, Any}("name" => "strategy_nominal", "data_type" => Any[], "default_value" => "\"most_frequent\"", "description" => Any[]), Dict{String, Any}("name" => "verbose", "data_type" => Any[], "default_value" => "0", "description" => Any[])], "class_name" => "openmlstudy14.preprocessing.ConditionalImputer", "external_version" => "openml==0.6.0,openmlstudy14==0.0.1", "uploader" => "1", "tag" => Any["openml-python", "python", "scikit-learn", "sklearn", "sklearn_0.18.1"], "upload_date" => "2017-11-22T10:40:20"))

