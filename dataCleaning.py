#!/usr/bin/python3

import numpy as np
import pandas as pd
import time, requests
from json import loads
from datetime import datetime
from sklearn.impute import SimpleImputer

nan_ratio_threshold = 0.97
yelp_catagories_json_url = "https://www.yelp.com/developers/documentation/v3/all_category_list/categories.json"

################################## Common ######################################
# Append k to list l n times
def fillList(l, k, n):
    return l + [k] * n

# Go through each feature col and print (#nan / #observation)
def printNanRatio(df, description):
    col_count = df.shape[0]
    print(f"--- Nan ratio for each feature of {description} ---")
    for col in df:
        curr_col = df[col]
        ratio = float(len(curr_col) - curr_col.count()) / col_count
        if (ratio > nan_ratio_threshold):
            print(f"(>{nan_ratio_threshold})\t", end ="")
        print(f"{col}: {ratio}")
    print("----------------------------------------------------")

# How many non nan observation where X and Y value are different
def compareFeaturesXY(df, x, y):
    x_col = df[x]
    y_col = df[y]
    diff = 0
    for x_value, y_value in zip(x_col, y_col):
        if (not pd.isnull(x_value) and not pd.isnull(y_value) and x_value != y_value):
            diff = diff + 1
    return diff

# Format string to be loadable by json.loads()
def formatJson(json_string):
    if not pd.isnull(json_string):
        return (json_string.replace('\'', '\"').replace(": ", ": \"").
                replace(',', "\",").replace('}', "\"}"))
    else:
        return json_string
    
# Fill missing data using sklearn Imputer for the whole df
def imputeDf(df, strategy, nan_value=np.nan):
    imp = SimpleImputer(strategy=strategy, missing_values=nan_value)
    imputed = pd.DataFrame(imp.fit_transform(df))
    imputed.columns = df.columns
    imputed.index = df.index
    return imputed

# Fill missing data using sklearn Imputer for df col
def imputeDfCol(df, colName, strategy, nan_value=np.nan, fill_value=None):
    imp = SimpleImputer(strategy=strategy, missing_values=nan_value, 
                        fill_value=fill_value)
    imputed = imp.fit_transform(pd.DataFrame(df[colName]))
    return imputed

# Categorical value to numerical value
def quantifyDf(df, colName, mapping_dict, filling_strategy):
    df[colName] = df[colName].astype(str)
    df[colName] = df[colName].str.strip()
    keys = mapping_dict.keys()
    for k in keys:
        df.loc[df[colName] == k, colName] = mapping_dict[k]
    df[colName] = df[colName].astype(float)
    df[colName] = imputeDfCol(df, colName, filling_strategy)
################################################################################

####################### Business data specific #################################
# Return new dataframe with cols extracted from "json" value of df[colName]
def handleJsonAttr(df, colName, filling_strategy, attr_prefix=""):
    # Find the longest value to make sure optional attr is incl
    longest_idx = df[colName].str.len().nlargest(1).keys()[0]
    attr_json = formatJson(df[colName][int(longest_idx)])
    parsed_json = loads(attr_json)
    
    # Creating attribute dict
    attr_dict = {}
    for attr in parsed_json:
        attr_dict.update({attr_prefix + attr : []})
        
    # Appending attribute value to dict
    for row in df[colName]:
        if (pd.isnull(row)):
            for k in attr_dict.keys():
                attr_dict[k].append(np.nan)
        else:
            parsed_json = loads(formatJson(row))
            for k in attr_dict.keys():
                ori_attr = k
                if (attr_prefix != ""):
                    ori_attr = k.split(attr_prefix)[1]
                value = 1 if (ori_attr in parsed_json.keys() 
                                and parsed_json[ori_attr] == "True") else 0
                attr_dict[k].append(value)
    
    # Df containing new attr and merge with the original dataframe on same index
    sub_df = pd.DataFrame(data=attr_dict)
    df = df.merge(sub_df, how="outer", left_index=True, right_index=True)
    df = df.drop(colName, axis=1)
    
    # Fill missing data for new attr
    for attr in attr_dict.keys():
        df[attr] = imputeDfCol(df, attr, filling_strategy)
    return df

# Return new dataframe with col indicating whether parking is available
def handleParkingAttr(df, parking_col, new_attr, filling_strategy):
    values = []
    for row in df[parking_col]:
        if pd.isnull(row):
            values.append(np.nan)
        elif "True" in row:
            values.append(1)
        else:
            values.append(0)
    sub_df = pd.DataFrame(values, columns=[new_attr])
    df = df.merge(sub_df, how="outer", left_index=True, right_index=True)
    df = df.drop(parking_col, axis=1)
    df[new_attr] = imputeDfCol(df, new_attr, filling_strategy)
    return df

# Return new dataframe with categories_col replaced by multiple parent category cols
def handleCategoriesAttr(df, categories_col):
    try:
        categories_json = loads(requests.get(yelp_catagories_json_url).text)
        categories_dict = {c['title']: c['parents'] for c in categories_json}
        alias_dict = {c['title']: c['alias'] for c in categories_json}
        alias_parents_dict = {c['alias']: c['parents'] for c in categories_json}
        present_categories = {}
        row_num = len(df[categories_col])
        
        for i, row in enumerate(df[categories_col]):
            row_categories = [] if pd.isnull(row) else row.split(', ')
            for c in row_categories:
                if c in categories_dict:
                    categories = []
                    if not categories_dict[c]:
                        categories.append(alias_dict[c])
                    else:
                        # Use grandparent to prevent the cases that one category
                        # only applies to one business/observation
                        categories = ([alias_parents_dict[x][0] if alias_parents_dict[x] 
                                            else x for x in categories_dict[c]])
                    for category in categories:
                        if category not in present_categories:
                            # Prev rows don't have this category so should be 0s
                            values = [] if i == 0 else [0] * i
                            # Current row
                            values.append(1)
                            present_categories[category] = values
                        else:
                            l = present_categories[category]
                            len_diff = i + 1 - len(l)
                            # No prev entry in row_categories already belongged 
                            # to current category
                            if len_diff >= 1:
                                if len_diff > 1:
                                    present_categories[category] = fillList(l, 0, len_diff-1) 
                                present_categories[category].append(1)
                            # Prev entry belongged to current category exists
                            elif len_diff == 0:
                                present_categories[category][i] += 1

        # The last row that had any category may not be the last row of column
        for c in present_categories:
            l = present_categories[c]
            len_diff = row_num - len(l)
            if len_diff > 0:
                present_categories[c] = fillList(l, 0, len_diff)

        sub_df = pd.DataFrame(data=present_categories)
        df = df.merge(sub_df, how="outer", left_index=True, right_index=True)
            
    except requests.ConnectionError:
        print("# Failed HTTP req for yelp catagories_json. Skipping categories")
    df = df.drop(categories_col, axis=1)
    return df

# Return new dataframe with new cols for each value in value_lst
def handleOrdinalAttr(df, colName, value_lst, filling_strategy):
    value_dict = {k: [] for k in value_lst}
    
    for row in df[colName]:
        if pd.isnull(row):
            [v.append(np.nan) for v in value_dict.values()]
        else:
            [v.append(1 if row == k else 0) for k, v in value_dict.items()]
    sub_df = pd.DataFrame(data=value_dict)
    df = df.merge(sub_df, how="outer", left_index=True, right_index=True)
    df = df.drop(colName, axis=1)
    
    # Fill missing data for new attr
    for attr in value_dict.keys():
        df[attr] = imputeDfCol(df, attr, filling_strategy)
    return df
    
# Return new dataframe with review counts replaced by the closest grouping counts
# helper for handleReviewCountBusiness() to find reasonable border
def handleReviewCountRange(df, rcount_col, start_count, count_range):
    max_count = df[rcount_col].nlargest(1).iloc[0]
    possible_counts = [c for c in range(start_count, max_count, count_range)]
    handled_counts = [possible_counts[np.argmin(
                        np.absolute(np.array(count) - possible_counts))] 
                        for count in df[rcount_col]]
    df[rcount_col] = handled_counts
    return df

# Return new dataframe with review_counts replaced with new attr indicating whether
# the business is heavily reviewd (review_count > border) 
def handleReviewCountBusiness(df, rcount_col, new_col, border):
    handled_counts = []
    main_col = df[rcount_col]
    for count in main_col:
        handled_counts.append(0 if count < border else 1)
    df = df.drop(rcount_col, axis=1)
    df[new_col] = handled_counts
    return df

def handleHours(df, hour_col):
    handled = []
    main_col = df[hour_col]
    for hr_range in main_col:
        if pd.isnull(hr_range):
            handled.append(0)
        else:
            start, end = [int(hr.split(':')[0]) for hr in hr_range.split('-')]
            hour_diff = (end-start) if (end > start) else (24-start+end)
            handled.append(hour_diff)
    df[hour_col] = handled
################################################################################

############################# User data specific ###############################
# Replace elite with number year being elite
def handleUserElite(df, elite_col):
    main_col = df[elite_col]
    handled = [(len(elite.split(',')) - (1 if elite=="None" else 0)) 
                for elite in main_col]
    df[elite_col] = handled
    
# Return new dataframe with yelping_since replaced with duration instead
def handleYelpingSince(df, ys_col, new_attr, unit="month"):
    divider = 365
    if unit != "year":
        divider = 30 if unit == "month" else 1
    now = datetime.now()
    handled = [(now - datetime.strptime(since, "%Y-%m-%d")).days / divider 
               for since in df[ys_col]]
    df[new_attr] = handled
    df = df.drop(ys_col, axis=1)
    return df

# Return a new dataframe with columns in target_col_list / review count column
def normalizeWithReviewCount(df, target_col_list, rcount_col):
    for target in target_col_list:
        df[target] = df[target] / df[rcount_col]
        df[target] = imputeDfCol(df, target, "constant", np.nan, 0)
    df = df.drop(rcount_col, axis=1)
    return df

# Normalize target_col by n
def normalizeWith(df, target_col, n):
    df[target_col] = df[target_col] / n
################################################################################
    
def cleanBusinessData(business_csv, cleaned_business_csv="business_cl.csv"):
    print("## Load", business_csv)
    bdf = pd.read_csv(business_csv)
    bstart = time.time()
    high_nan_attrs = ["attributes_AgesAllowed"]
    unused_attrs = ["address", "attributes", "hours", "name", "neighborhood", 
                    "city", "state", "postal_code"]
    clean_bdf = bdf.drop(high_nan_attrs + unused_attrs, axis=1)
    clean_bdf["attributes_RestaurantsPriceRange2"] = imputeDfCol(clean_bdf, 
                                     "attributes_RestaurantsPriceRange2", "most_frequent")
    # Specially handled attrs
    hour_attrs = ["hours_Monday", "hours_Tuesday", 
                    "hours_Wednesday", "hours_Thursday", "hours_Friday", 
                    "hours_Saturday", "hours_Sunday"]
    for attr in hour_attrs:
        handleHours(clean_bdf, attr)

    clean_bdf = handleParkingAttr(clean_bdf, "attributes_BusinessParking", 
                                                  "parking_avail", "most_frequent")
    clean_bdf = handleCategoriesAttr(clean_bdf, "categories")
    
    # Ordinal attrs
    clean_bdf = handleOrdinalAttr(clean_bdf, "attributes_RestaurantsAttire", 
                                  ["casual", "dressy"], "most_frequent")
    clean_bdf = handleOrdinalAttr(clean_bdf, "attributes_NoiseLevel",
                                  ["quiet", "average", "loud", "very_loud"], "most_frequent")
    
    # Seperate attr from json
    attrsWithJson = ["attributes_Ambience", "attributes_BestNights", 
                     "attributes_GoodForMeal", "attributes_Music", 
                     "attributes_HairSpecializesIn", 
                     "attributes_DietaryRestrictions"]
    attrsPrefix = ["atm_", "best_", "meal_", "mus_", "hair_", "dietary_"]
    for i, attr in enumerate(attrsWithJson):
        clean_bdf = handleJsonAttr(clean_bdf, attr, "most_frequent", attrsPrefix[i])
     
    # Quantify attr values
    attrWithOnlyBool = ["attributes_BikeParking", "attributes_BYOB",
                       "attributes_BusinessAcceptsBitcoin", 
                       "attributes_BusinessAcceptsCreditCards", 
                       "attributes_ByAppointmentOnly", "attributes_Caters", 
                       "attributes_CoatCheck", "attributes_Corkage", 
                       "attributes_DogsAllowed", "attributes_DriveThru", 
                       "attributes_GoodForDancing", "attributes_GoodForKids",
                       "attributes_HappyHour", "attributes_HasTV", 
                       "attributes_RestaurantsDelivery",
                       "attributes_RestaurantsGoodForGroups", 
                       "attributes_RestaurantsReservations", 
                       "attributes_RestaurantsTableService", 
                       "attributes_RestaurantsTakeOut",
                       "attributes_OutdoorSeating",
                       "attributes_WheelchairAccessible","attributes_AcceptsInsurance","attributes_Open24Hours", 
                      "attributes_RestaurantsCounterService"]

    for attr in attrWithOnlyBool:
        quantifyDf(clean_bdf, attr, {"True": 1.0, "False": 0.0}, "most_frequent")
    quantifyDf(clean_bdf, "attributes_Alcohol", 
               {"full_bar": 1.0, "beer_and_wine": 1.0, "none": 0.0}, "most_frequent")
    quantifyDf(clean_bdf, "attributes_BYOBCorkage", 
               {"yes_free": 1.0, "yes_corkage": 1.0, "no": 0.0}, "most_frequent")
    quantifyDf(clean_bdf, "attributes_Smoking", 
               {"yes": 1.0, "outdoor": 1.0, "no": 0.0}, "most_frequent")
    quantifyDf(clean_bdf, "attributes_WiFi", 
               {"free": 1.0, "paid": 0.75, "no": 0.0}, "most_frequent")
    
    bend = time.time()
    print("## Write cleaned df to", cleaned_business_csv)
    clean_bdf.to_csv(cleaned_business_csv, index=False)
    print(f"## Took: {bend - bstart} sec")

def cleanUsersData(users_csv, cleaned_users_csv):
    print("## Load", users_csv)
    udf = pd.read_csv("users.csv")
    ustart = time.time()
    unused_attrs = ["name", "friends"]
    clean_udf = udf.drop(unused_attrs, axis=1)
    clean_udf = handleYelpingSince(clean_udf, "yelping_since", "yelping_for")
    handleUserElite(clean_udf, "elite")
    
    normalize_target_cols = ["compliment_cool", "compliment_cute", 
                             "compliment_funny", "compliment_hot", 
                             "compliment_list", "compliment_more", 
                             "compliment_note", "compliment_photos", 
                             "compliment_plain", "useful", "compliment_profile",
                             "compliment_writer", "cool", "funny"]

    uend = time.time()
    print("## Write cleaned df to", cleaned_users_csv)
    clean_udf.to_csv(cleaned_users_csv, index=False)
    print(f"## Took: {uend - ustart} sec")
