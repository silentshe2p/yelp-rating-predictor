import pandas as pd
import time
        
def populate_combined_review(users_df, business_df, reviews_df):
    sub_df = reviews_df.drop(["cool", "funny", "useful"], axis=1, errors="ignore")
    output_df = sub_df.merge(users_df, how="left", on="user_id").drop(["user_id"], axis=1)
    output_df = output_df.merge(business_df, how="left", on="business_id").drop(["business_id"], axis=1)
    return output_df

def combine_data(users_csv, business_csv, train_reviews_csv, validate_queries_csv, test_queries_csv):
    start = time.time()
    print("## Load csv")
    users_df = pd.read_csv(users_csv)
    business_df = pd.read_csv(business_csv)
    
    train_reviews_df = pd.read_csv(train_reviews_csv).drop(["date", "text", "review_id"], axis=1)
    train_reviews_stars = train_reviews_df["stars"]
    train_reviews_df = train_reviews_df.drop(["stars"], axis=1)
    
    train_test_reviews_df = pd.read_csv(validate_queries_csv)
    train_test_reviews_stars = train_test_reviews_df["stars"]
    train_test_reviews_df = train_test_reviews_df.drop(["stars"], axis=1)
    
    test_reviews_df = pd.read_csv(test_queries_csv)

    print("## Populate combined train reviews")
    combined_train_review = populate_combined_review(users_df, business_df, train_reviews_df)
    combined_train_review["review_stars"] = train_reviews_stars
    print("## Populate combined train test reviews")
    combined_train_test_review = populate_combined_review(users_df, business_df, train_test_reviews_df)
    combined_train_test_review["review_stars"] = train_test_reviews_stars
    print("## Populate combined test reviews")
    combined_test_review = populate_combined_review(users_df, business_df, test_reviews_df)
    
    print("## Write to csv")   
    combined_train_review.to_csv("train_reviews_combined.csv", index=False)
    combined_train_test_review.to_csv("validate_queries_combined.csv", index=False)
    combined_test_review.to_csv("test_queries_combined.csv", index=False)
    
    end = time.time()
    print(f"## Took: {end - start} sec")
    