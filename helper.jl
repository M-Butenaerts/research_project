using HTTP, JSON, DataFrames, OpenML, CSV

function get_info(dataset_id::Int64) 

    # Get the task associated with the dataset ID
    url = "https://www.openml.org/api/v1/json/task/list/data_id/$(dataset_id)"
    response = HTTP.get(url)
    task_data = JSON.parse(String(response.body))["tasks"]["task"]
    task_id = task_data[1]["task_id"]
    

    evaluation_url = "https://www.openml.org/api/v1/json/evaluation/list/function/area_under_roc_curve/task/$task_id/sort_order/desc/limit/1"
    evaluation_response = HTTP.get(evaluation_url)
    evaluation_data = JSON.parse(String(evaluation_response.body))["evaluations"]["evaluation"][1]

    # best_run_id = evaluation_data["run_id"]

    flow_id = evaluation_data["flow_id"]
    flow_url = "https://www.openml.org/api/v1/json/flow/$(flow_id)"
    flow_response = HTTP.get(flow_url)
    flow_data = JSON.parse(String(flow_response.body))["flow"] 


    function summarize_component(component_input::Pair{String, Any})
        component_info = component_input.second

        println("       Parameters:")

        if isa(component_info["parameter"], Dict{String, Any})
            print("           Name: $(component_info["parameter"]["name"])")
            println("  Data Type: ", (component_info["parameter"]["data_type"]))
        elseif !isempty(component_info["parameter"])
            for parameter in component_info["parameter"]
                print("           Name: $(parameter["name"])")
                println("  Default Value: ", isempty(parameter["default_value"]) ? "None" : parameter["default_value"])            
            end
        else
            println("   No parameters.")
        end
        println()
    end

    function display_flow_info(flow_info)
        println("Name: $(flow_info["name"])")
        println("Data name:", task_data[1]["name"])
        println("$(evaluation_data["function"]): $(evaluation_data["value"])")

        if (haskey(flow_info, "custom_name"))
            println("Custom name: $(flow_info["custom_name"])")
        end
        println()
        println("Description: $(flow_info["description"])")

        println()
        println("Components:")
        for (component, value) in flow_info["component"]
            name = component.second

            println("   Component: $(name)")
            summarize_component(value)
        end

    end

    display_flow_info(flow_data)
end


function get_dataset(dataset_id::Int64)
    # Get the data itself as a dataframe
    table = OpenML.load(dataset_id)
    df = DataFrame(table)
    return df
end

function get_dataset_info(dataset_id::Int64)
    # Get dataset by ID
    OpenML.describe_dataset(dataset_id)
end

function save_dataset_to_csv(filename::String, df::DataFrame)
    path = dirname(pwd()) * "\\research_project\\Datasets\\" * filename * ".csv"
    CSV.write(path, df)
end

function read_dataset_from_csv(filename::String)
    path = dirname(pwd()) * "\\research_project\\Datasets\\" * filename * ".csv"
    df = DataFrame(CSV.File(path))
    return df
end