import random

num_texts = 60000

file = open("texts.txt",'a+')
for i in range(num_texts):
	cheque_number = random.randint(0,999999)
	cheque_string = "0"*(6-len(str(cheque_number)))+str(cheque_number)

	micr_code = random.randint(0,999999999)
	micr_string = "0"*(9-len(str(micr_code)))+str(micr_code)

	account_id = random.randint(0,999999)
	account_id_string = "0"*(6-len(str(account_id)))+str(account_id)
	text = f"c{cheque_string}c {micr_string}a {account_id_string}c 31\n"

	file.write(text)
file.close()