log=logs/18_04_24/
file="PROTEINS"
#"ENZYMES"FRANKENSTEIN""PROTEINS" 
for dataset in  "PROTEINS" 
do
python3 main.py  --logging $log$file --pooling lspool --dataset $dataset "
done
