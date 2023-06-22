
# import experiments
import text_attention
import prompt_mapping
import noise_mapping
import proof_of_concept



# attention plots
text_attention.run_attention_plots()

# manipulate embeddings
text_attention.run_manipulate_embeddings()



# perturbation mapping
prompt_mapping.run_perturbation_mapping()



# common sense bar plot
noise_mapping.run_common_sense()



# joystick prompt run
proof_of_concept.joystick_prompt_run()