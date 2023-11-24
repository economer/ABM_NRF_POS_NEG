from piplyr import piplyr
from run import df



(
piplyr(df).
drop_col('Step').
sql_plyr("""
SELECT
    step,
    age_sex,
    AVG(biweekly_income) as mean_income,
    AVG(nrf) AS mean_nrf,
    AVG(pos) AS mean_pos,
    AVG(neg) AS mean_neg
FROM
    df
GROUP BY
    step,
    age_sex
    
""").
sql_plyr("""
         select *
         from df
         where 
         age_sex == 'male_50+'
        
         """).to_df

)