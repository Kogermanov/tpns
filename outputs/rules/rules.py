def findDecision(obj): #obj[0]: LIMIT_BAL, obj[1]: SEX, obj[2]: EDUCATION, obj[3]: MARRIAGE, obj[4]: AGE, obj[5]: PAY_0, obj[6]: PAY_2, obj[7]: PAY_3, obj[8]: PAY_4, obj[9]: PAY_5, obj[10]: PAY_6, obj[11]: BILL_AMT1, obj[12]: PAY_AMT1, obj[13]: PAY_AMT2, obj[14]: PAY_AMT3, obj[15]: PAY_AMT4, obj[16]: PAY_AMT5, obj[17]: PAY_AMT6
   # {"feature": "PAY_0", "instances": 30000, "metric_value": 0.0346, "depth": 1}
   if obj[5]<=0:
      # {"feature": "PAY_3", "instances": 23182, "metric_value": 0.0039, "depth": 2}
      if obj[7]<=0:
         # {"feature": "PAY_AMT3", "instances": 21807, "metric_value": 0.0025, "depth": 3}
         if obj[14]>0:
            # {"feature": "LIMIT_BAL", "instances": 18574, "metric_value": 0.0025, "depth": 4}
            if obj[0]<=182893.48982448585:
               # {"feature": "PAY_AMT2", "instances": 10687, "metric_value": 0.0021, "depth": 5}
               if obj[13]<=4524.738280153458:
                  return 0.15036601464058563
               elif obj[13]>4524.738280153458:
                  return 0.09048428207306712
               else:
                  return 0.13717600823430337
            elif obj[0]>182893.48982448585:
               # {"feature": "BILL_AMT1", "instances": 7887, "metric_value": 0.0015, "depth": 5}
               if obj[11]<=296255.08734484063:
                  return 0.08184245660881175
               elif obj[11]>296255.08734484063:
                  return 0.16876574307304787
               else:
                  return 0.08621782680360086
            else:
               return 0.11553784860557768
         elif obj[14]<=0:
            # {"feature": "PAY_AMT1", "instances": 3233, "metric_value": 0.0022, "depth": 4}
            if obj[12]<=3092.9167955459325:
               # {"feature": "PAY_5", "instances": 2565, "metric_value": 0.0019, "depth": 5}
               if obj[9]<=0:
                  return 0.20714865962632006
               elif obj[9]>0:
                  return 0.3883495145631068
               else:
                  return 0.21442495126705652
            elif obj[12]>3092.9167955459325:
               # {"feature": "LIMIT_BAL", "instances": 668, "metric_value": 0.01, "depth": 5}
               if obj[0]<=431992.28618830396:
                  return 0.14330708661417324
               elif obj[0]>431992.28618830396:
                  return 0
               else:
                  return 0.13622754491017963
            else:
               return 0.19826786266625426
         else:
            return 0.12780299903700645
      elif obj[7]>0:
         # {"feature": "PAY_AMT4", "instances": 1375, "metric_value": 0.0051, "depth": 3}
         if obj[15]>0:
            # {"feature": "PAY_AMT6", "instances": 994, "metric_value": 0.004, "depth": 4}
            if obj[17]<=31802.31733673017:
               # {"feature": "PAY_2", "instances": 981, "metric_value": 0.0033, "depth": 5}
               if obj[6]<=2:
                  return 0.259726603575184
               elif obj[6]>2:
                  return 0.5666666666666667
               else:
                  return 0.2691131498470948
            elif obj[17]>31802.31733673017:
               return 0
            else:
               return 0.2655935613682093
         elif obj[15]<=0:
            # {"feature": "PAY_AMT1", "instances": 381, "metric_value": 0.0088, "depth": 4}
            if obj[12]<=23963.48207318998:
               # {"feature": "LIMIT_BAL", "instances": 373, "metric_value": 0.0086, "depth": 5}
               if obj[0]<=302837.91281416925:
                  return 0.4375
               elif obj[0]>302837.91281416925:
                  return 0.09523809523809523
               else:
                  return 0.41823056300268097
            elif obj[12]>23963.48207318998:
               return 0
            else:
               return 0.4094488188976378
         else:
            return 0.3054545454545455
      else:
         return 0.13834009145026313
   elif obj[5]>0:
      # {"feature": "PAY_2", "instances": 6818, "metric_value": 0.0229, "depth": 2}
      if obj[6]>-1:
         # {"feature": "PAY_6", "instances": 4859, "metric_value": 0.0058, "depth": 3}
         if obj[10]<=0:
            # {"feature": "PAY_AMT1", "instances": 3031, "metric_value": 0.0028, "depth": 4}
            if obj[12]<=2261.061365885846:
               # {"feature": "PAY_AMT4", "instances": 2110, "metric_value": 0.0014, "depth": 5}
               if obj[15]<=25741.990592450136:
                  return 0.5093704949543488
               elif obj[15]>25741.990592450136:
                  return 0.20689655172413793
               else:
                  return 0.5052132701421801
            elif obj[12]>2261.061365885846:
               # {"feature": "PAY_3", "instances": 921, "metric_value": 0.0049, "depth": 5}
               if obj[7]>-2:
                  return 0.62472647702407
               elif obj[7]<=-2:
                  return 0
               else:
                  return 0.6199782844733985
            else:
               return 0.5400857802705378
         elif obj[10]>0:
            # {"feature": "PAY_AMT3", "instances": 1828, "metric_value": 0.0019, "depth": 4}
            if obj[14]<=12695.743907358257:
               # {"feature": "PAY_5", "instances": 1789, "metric_value": 0.0016, "depth": 5}
               if obj[9]>0:
                  return 0.7132401862940785
               elif obj[9]<=0:
                  return 0.6153846153846154
               else:
                  return 0.6975964225824483
            elif obj[14]>12695.743907358257:
               # {"feature": "AGE", "instances": 39, "metric_value": 0.0567, "depth": 5}
               if obj[4]>26.436777312013916:
                  return 0.47058823529411764
               elif obj[4]<=26.436777312013916:
                  return 0
               else:
                  return 0.41025641025641024
            else:
               return 0.6914660831509847
         else:
            return 0.597036427248405
      elif obj[6]<=-1:
         # {"feature": "PAY_AMT4", "instances": 1959, "metric_value": 0.0074, "depth": 3}
         if obj[15]<=0:
            # {"feature": "EDUCATION", "instances": 1155, "metric_value": 0.0043, "depth": 4}
            if obj[2]<=3:
               # {"feature": "AGE", "instances": 1141, "metric_value": 0.004, "depth": 5}
               if obj[4]<=54.33945062647264:
                  return 0.3193046660567246
               elif obj[4]>54.33945062647264:
                  return 0.625
               else:
                  return 0.33216476774758985
            elif obj[2]>3:
               return 0
            else:
               return 0.3281385281385281
         elif obj[15]>0:
            # {"feature": "PAY_4", "instances": 804, "metric_value": 0.0034, "depth": 4}
            if obj[8]<=0:
               # {"feature": "EDUCATION", "instances": 762, "metric_value": 0.003, "depth": 5}
               if obj[2]<=3:
                  return 0.1768617021276596
               elif obj[2]>3:
                  return 0
               else:
                  return 0.17454068241469817
            elif obj[8]>0:
               # {"feature": "LIMIT_BAL", "instances": 42, "metric_value": 0.0479, "depth": 5}
               if obj[0]<=441087.66670398484:
                  return 0.3333333333333333
               elif obj[0]>441087.66670398484:
                  return 1
               else:
                  return 0.38095238095238093
            else:
               return 0.1853233830845771
         else:
            return 0.26952526799387444
      else:
         return 0.5029334115576415
   else:
      return 0.2212
