INPUT=../data/
OUTPUT=../output/
export CUDA_VISIBLE_DEVICES=1

for model in t5 pubmedgpt
do
    python ppl_score.py \
            --hypo_path $INPUT \
            --hypo_file simplification_en_de_en_gpt3_perturbation.csv \
            --output_path $INPUT \
            --model $model
done

python pomme.py \
        --ref_input_path ../data/ref/ \
        --indomain_model pubmedgpt \
        --outdomain_model t5 \
        --ref_prefix_src source \
        --ref_prefix_tgt target \
        --hypo_input_path $INPUT \
        --hypo_prefix simplification_en_de_en_gpt3_perturbation \
        --hypo_output_path $OUTPUT \
        --ppl_name perplexity_score