import pandas as pd
import ast

def main():
    
    # Extract data required for finetuning LLMs
    # Save into split_type_extracted.csv
    
    extract_gameplan_data('train')
    extract_gameplan_data('valid_seen')
    
    # Create the dataset required for Gameplan-prediciton
    # Dialog History (DH), DH + Dialog Act information (DH + DA), DH + DA + Filter (Only <<Instruction>> type)
    create_gameplan_dataset('train')
    create_gameplan_dataset('valid_seen')

    
    return None

def extract_gameplan_data(split_type):
    gameplan_df = pd.read_pickle('../../project/TEACh_PME/data/TEACh_dataset_main_' + split_type + '.pkl')
    drop_cols = ['dialog_history', 'driver_action_history',
       'driver_image_history', 'driver_actions_future', 'driver_images_future',
       'interactions', 'game_id', 'instance_id', 'pred_start_idx',
       'init_state_diff', 'final_state_diff', 'state_changes',
       'history_subgoals', 'future_subgoals',
       'expected_init_goal_conditions_total',
       'expected_init_goal_conditions_satisfied', 'dialog_history_cleaned',
       'dialog_history_proc', 'driver_actions_proc',
       'source_actions', 'target_actions', 'target_obj_interaction_actions',
       'target_objects', 'driver_images_history_feats',
       'driver_images_future_feats', 'driver_interactive_index_history',
       'driver_interactive_indices_future', 'game_plan_future_indices']
    
    df = gameplan_df.drop(columns = drop_cols, axis=1)
    df = df.loc[:, ~df.columns.str.match('Unnamed')]
    df.to_csv('llms/gameplan_data/' + split_type + '_extracted.csv')

    
def create_gameplan_dataset(split_type):
    
    # 4258 length
    df = pd.read_csv('llms/gameplan_data/' + split_type + '_extracted.csv')
    
    # Only Dialog History
    dh = []    
    for i in range(len(df)):
        dialog = df['processed_dialog_history'].iloc[i]
        proc_dialog = dialog.replace("<<Commander>>", "").replace("<<Driver>>", "")
        dh.append(proc_dialog)
        
    # Dialog History with Dialog Act Information
    dh_da = df['processed_dialog_history_with_das']
    
    # DH + DA + Filter 
    dh_da_f = []
    for i in range(len(df)):
        str_dh = df['dialog_history_with_das'].iloc[i]
        list_dh = ast.literal_eval(str_dh)
        
        output = ''
        for item in list_dh:
            if '<<Instruction>>' in item[1]:
                output += "<<"+item[0]+">> "+item[1]+" "
                
        dh_da_f.append(output)
        
    
    # Gameplan Prediction (History + Future)
    gameplan_prediction = []
    for i in range(len(df)):
        str_gph = df['game_plan_history'].iloc[i]
        str_gpf = df['game_plan_future'].iloc[i]
        list_gph = ast.literal_eval(str_gph)
        list_gpf = ast.literal_eval(str_gpf)
        
        output = ''
        for gph in list_gph:
            output += gph
            output += ' -- '
            
        for k in range(len(list_gpf)):
            output += list_gpf[k]
            if k != len(list_gpf)-1:
                output += ' -- '
       
        gameplan_prediction.append(output)
         
        
    data_dh = {
        "dialog" : dh,
        "gameplan_prediction": gameplan_prediction
    }
    
    data_da = {
        "dialog" : dh_da,
        "gameplan_prediction": gameplan_prediction
    }
    
    data_f = {
        "dialog" : dh_da_f,
        "gameplan_prediction": gameplan_prediction
    }
    
  
    pd.DataFrame(data_dh).to_csv('llms/gameplan_data/' + split_type + '_dh.csv')
    pd.DataFrame(data_da).to_csv('llms/gameplan_data/' + split_type + '_dh_da.csv')
    pd.DataFrame(data_f).to_csv('llms/gameplan_data/' + split_type + '_dh_da_f.csv')
    

if __name__ == '__main__':
    main()