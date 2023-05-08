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


function display_flow_info(flow_info)
    println("Name: $(flow_info["name"])")
    if (haskey(flow_info, "custom_name"))
        println("Custom name: $(flow_info["custom_name"])")
    end
    println("Description: $(flow_info["description"])")

    println("Parameters:")
    for (parameter, value) in flow_info["parameter"]
        println("  $(parameter): $(value)")
    end

    println("Components:")
    for (component, value) in flow_info["components"]
        println("  $(component): $(value)")
    end

end

display_flow_info(flow_data)