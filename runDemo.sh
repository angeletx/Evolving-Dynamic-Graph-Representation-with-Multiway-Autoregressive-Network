python run_exp.py --config_file "./config/email_eu_dept1/parameter(dataset=email_eu_dept1,task=link_pred,model=egcn_man,learningRate=0.001000,historyStep=5,gcnHiddenFeature=[64x20]).yaml"
python log_analyzer.py "./blog/email_eu_dept1/dataset=email_eu_dept1,task=link_pred,model=egcn_man,learningRate=0.001000,historyStep=5,gcnHiddenFeature=[64x20].log"

python run_exp.py --config_file "./config/email_eu_dept1/parameter(dataset=email_eu_dept1,task=link_pred,model=gcn_man,learningRate=0.001000,historyStep=5,gcnHiddenFeature=[64x20]).yaml"
python log_analyzer.py "./blog/email_eu_dept1/dataset=email_eu_dept1,task=link_pred,model=gcn_man,learningRate=0.001000,historyStep=5,gcnHiddenFeature=[64x20].log"

