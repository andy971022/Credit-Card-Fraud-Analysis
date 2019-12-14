import pandas as pd 
import pandas_profiling

def make_profiling(df,name):
	profile = df.profile_report(
		check_correlation_pearson = False,
		correlations = {
		'spearman': False,
		'kendall': False,
		'phi_k': False,
		'cramers': False,
		'recoded': False,
		}
		)
	profile.to_file(output_file = f"./EDA/{name}.html")	


if __name__ == '__main__':
	df = pd.read_csv("./CSV_Files/creditcard.csv")
	print(df.columns)
	neg_df = df[df["Class"] == 0].sample(10000)
	pos_df = df[df["Class"] == 1]
	df = pd.concat([neg_df,pos_df])

	make_profiling(neg_df,"neg_eda")
	make_profiling(pos_df,"pos_eda")
	make_profiling(df,"all_eda")
