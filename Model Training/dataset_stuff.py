from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
letter_recognition = fetch_ucirepo(id=59) 
  
# data (as pandas dataframes) 
X = letter_recognition.data.features 
y = letter_recognition.data.targets 

# metadata 
print(letter_recognition.metadata) 
  
# variable information 
print(letter_recognition.variables) 

# Save features
X.to_csv("letter_recognition_features.csv", index=False)

# Save targets
y.to_csv("letter_recognition_targets.csv", index=False)