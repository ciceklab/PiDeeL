import sys
sys.path.insert(1, "../../")
from load_targeted_data import predicted_quant


cols_stand =['2-hydroxyglutarate', '3-hydroxybutyrate', 'Acetate',\
       'Alanine', 'Allocystathionine', 'Arginine', 'Ascorbate',\
       'Aspartate', 'Betaine', 'Choline', 'Creatine',\
       'Ethanolamine', 'GABA', 'Glutamate', 'Glutamine', 'Glutahionine (GSH)',\
       'Glycerophosphocholine', 'Glycine', 'hypo-Taurine',\
       'Isoleucine', 'Lactate', 'Leucine', 'Lysine', 'Methionine',\
       'myo-Inositol', 'NAL', 'NAA', 'o-Acetylcholine', 'Ornithine',\
       'Phosphocholine', 'Phosphocreatine', 'Proline',\
       'scyllo-Inositol', 'Serine', 'Taurine', 'Threonine',\
       'Valine']
from matplotlib import pyplot as plt
#draw a bar plot for 0th index sample of predicted_quant
plt.figure(figsize=(20,10))
plt.bar(cols_stand,predicted_quant[0])
plt.xticks(rotation=90,fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel("Concentration",fontsize=20)
plt.xlabel("Metabolites",fontsize=20)
plt.tight_layout()
plt.savefig("ab.png")
