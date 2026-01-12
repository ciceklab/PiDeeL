import pdb
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




most_meta = ["Glutamine", "Glutamate", "Alanine", "Aspartate", "Serine", "Glycine", "Lactate", "Methionine", "Ornithine", "Arginine", "Leucine", "GABA", "Glutahionine (GSH)", "Valine","Other metabolites"]
most_meta = most_meta[::-1]
most_pathways = ["Mineral absorption", "Metabolic pathways", "Alanine| aspartate and glutamate metabolism", "mTOR signaling pathway", "Central carbon metabolism in cancer", "Protein digestion and absorption", "Taurine and hypotaurine metabolism", "Glycine| serine and threonine metabolism", "Biosynthesis of cofactors", "Arginine biosynthesis", "Aminoacyl-tRNA biosynthesis", "Neuroactive ligand-receptor interaction", "Biosynthesis of amino acids", "Glutathione metabolism", "Butanoate metabolism","Other pathways"]
most_pathways = most_pathways[::-1]


abb = {"Glutamine" : 100.0,
"Glutamate" :75.08555615174568,
"Alanine" :64.96182152833359,
"Aspartate": 60.17144623097622,
"Serine" :54.618792068106906,
"Glycine" :53.85238806813412,
"Lactate": 51.59836023655551,
"Methionine" :50.14367554569308,
"Ornithine" :48.25138968001829,
"Arginine" :45.83668256796922,
"Leucine" :45.819036801335244,
"GABA" :45.10358347023839,
"Glutahionine (GSH)" :44.84466305908695,
"Valine" :40.68224261523055,
"Other metabolites" : 314.5230476546876 
}


baa = {
"Mineral absorption":96.09637969874039,
"Metabolic pathways" :86.10641238162161,
"Alanine| aspartate and glutamate metabolism":77.29429527213094,
"mTOR signaling pathway" :69.29096039303866,
"Central carbon metabolism in cancer" :62.12818878279817,
"Protein digestion and absorption" :54.73395885095679,
"Taurine and hypotaurine metabolism" :52.372747941005926,
"Glycine| serine and threonine metabolism" :50.20783459991536,
"Biosynthesis of cofactors" :49.654981694952525,
"Arginine biosynthesis" :49.58161104109674,
"Aminoacyl-tRNA biosynthesis" :48.523341563509604,
"Neuroactive ligand-receptor interaction" :47.236671016082575,
"Biosynthesis of amino acids" :47.20205976889019,
"Glutathione metabolism" :43.82218156829632,
"Butanoate metabolism" :43.18538259848382,
"Other pathways" : 2456.393289214002
}
"""
0,Alanine,Residual,1.5567407316561628
1,Glutamine,Residual,2.396393289214002
2,Glycine,Residual,1.3088810677698635
3,Isoleucine,Residual,1.0808592474758285
4,Methionine,Residual,1.2905150137462478
5,Serine,Residual,1.0980043230896916
6,Threonine,Residual,0.9749065319331437
7,Valine,Residual,1.0746544961185902
8,Others,Residual,12.329995785456296
"""
j = 0

#most_meta = most_meta[::-1]
#for item in most_meta:

    #print(str(j) + "," + str(item) + "," + str("Residual") + "," + str((a[item]*b["Residual"]**5)**0.01))
    #j += 1


with open("/home/gunkaynar/targeted_survival/scripts/result.csv") as f:
    lines = f.readlines()
    pathways = lines[0].split(",")
    for k in range(len(most_pathways)):
        if most_pathways[k] != "Other pathways":
            index = pathways.index(most_pathways[k])
            meta_interest = []
            for i in range(1,len(lines)):
                line = lines[i].split(",")
                if line[index] == "1" and cols_stand[i-1] in most_meta:
                    meta_interest.append(cols_stand[i-1])
            meta_interest  = [x for x in most_meta if x in meta_interest]
            meta_interest.append("Other metabolites")
        else:
            continue
        to = 0

        for item in meta_interest:
            
            to += abb[item]
        for item in meta_interest:
            print(str(j) + "," + str(item) + "," + str(most_pathways[k]) + "," + str(abb[item]/to * baa[most_pathways[k]]/10)+"," + str(abb[item]/to * baa[most_pathways[k]]/10))
            j+=1



print(list(baa.keys()))


