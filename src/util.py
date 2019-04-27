import sys

from typing import List

base_img_url = "https://images.dpchallenge.com/images_challenge"

def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s .%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()
    
def get_challenge_range(challenge:int) -> str:
    if challenge < 1000:
        return "0-999"
    elif challenge < 2000:
        return "1000-1999"
    elif challenge < 3000:
        return "2000-3999"

def get_link(img_id:str, challanger:str) -> str:
    return "{0}/{1}/{2}/1200/Copyrighted_Image_Reuse_Prohibited_{3}.jpg".format(
        base_img_url, get_challenge_range(int(challanger)), int(challanger), img_id)
    
def get_links_from_file(path:str, batches:int, epoch:int) -> List[str]:
    url_list = []
    print("Dataset setting up epoch: {0}".format(epoch))
    with open(path, "r") as file:
        item = 0
        for line in file.readlines()[epoch*batches:(epoch+1)*batches]:
            values = line.split(" ")
            ulr = get_link(img_id=values[1], challanger=values[14])
            url_list.append(ulr)
            item += 1
            progress(item, batches, status="Getting links")
            if item == batches:
                return url_list
                