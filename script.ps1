$DATASET_PATH = "C:\\Users\\Eusfo\\Desktop\\shrec2022-geometric-primitives\\data"
$BIM_TYPE = $args[0]

$files = Get-ChildItem -Path $DATASET_PATH -Recurse -Filter "*.txt"

foreach ($file in $files) {
    python evaluation.py --file=$($file.FullName) --outf=".\\output" --type=$BIM_TYPE
}
