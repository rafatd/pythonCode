def   substring(str1, str2):
            if   len(str1)==0 or len(str2)==0:
                 returnÂ False

           if str1[0: len(str2)]==str2:
              return True
           else:
              returnÂ substring(str1[1:],Â str2)
