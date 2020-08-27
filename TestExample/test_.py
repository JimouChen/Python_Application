from bs4 import BeautifulSoup

resp = """


<ul class="col5">

    <li class="clearfix">
        <span class="green-num-box">1</span>
        <a class="face" href="https://site.douban.com/SakyoStan/" target="_blank">
            <img src="https://img1.doubanio.com/view/site/small/public/1ba1de62fa91cdb.jpg">
        </a>
        <div class="intro">
            <h3 class="icon-play" data-sid="760149">
              <a href="javascript:;">Aug 11 お盆</a>
            </h3>

            <p>SakyoStan&nbsp;/&nbsp;690次播放</p>
        </div>
        <span class="days">(上榜3天)</span>
        <span class="trend arrow-up"> 1 </span>
    </li>

    <li class="clearfix">
        <span class="green-num-box">2</span>
        <a class="face" href="https://site.douban.com/echen/" target="_blank">
            <img src="https://img1.doubanio.com/view/site/small/public/bef1ba170d1e80b.jpg">
        </a>
        <div class="intro">
            <h3 class="icon-play" data-sid="760155">
              <a href="javascript:;">Pasiwali</a>
            </h3>

            <p>程程&nbsp;/&nbsp;1764次播放</p>
        </div>
        <span class="days">(上榜13天)</span>
        <span class="trend arrow-down"> 1 </span>
    </li>

    <li class="clearfix">
        <span class="green-num-box">3</span>
        <a class="face" href="https://site.douban.com/chameleon/" target="_blank">
            <img src="https://img3.doubanio.com/view/site/small/public/78207b30d57369f.jpg">
        </a>
        <div class="intro">
            <h3 class="icon-play" data-sid="760241">
              <a href="javascript:;">hurt people hurt people</a>
            </h3>

            <p>chameleon&nbsp;/&nbsp;385次播放</p>
        </div>
        <span class="days">(上榜6天)</span>
        <span class="trend arrow-up"> 7 </span>
    </li>

    <li class="clearfix">
        <span class="green-num-box">4</span>
        <a class="face" href="https://site.douban.com/rockeryu/" target="_blank">
            <img src="https://img1.doubanio.com/view/site/small/public/a70ca0cba66a62c.jpg">
        </a>
        <div class="intro">
            <h3 class="icon-play" data-sid="760142">
              <a href="javascript:;">Memory of mir</a>
            </h3>

            <p>于师傅&nbsp;/&nbsp;1554次播放</p>
        </div>
        <span class="days">(上榜15天)</span>
        <span class="trend arrow-down"> 1 </span>
    </li>

    <li class="clearfix">
        <span class="green-num-box">5</span>
        <a class="face" href="https://site.douban.com/caoyue/" target="_blank">
            <img src="https://img3.doubanio.com/view/site/small/public/8357739fb0e9ff2.jpg">
        </a>
        <div class="intro">
            <h3 class="icon-play" data-sid="759916">
              <a href="javascript:;">好demo</a>
            </h3>

            <p>曹悦&nbsp;/&nbsp;1437次播放</p>
        </div>
        <span class="days">(上榜15天)</span>
        <span class="trend arrow-down"> 1 </span>
    </li>

    <li class="clearfix">
        <span class="green-num-box">6</span>
        <a class="face" href="https://site.douban.com/huangyaowei/" target="_blank">
            <img src="https://img3.doubanio.com/view/site/small/public/8e3edc7ea7ec67f.jpg">
        </a>
        <div class="intro">
            <h3 class="icon-play" data-sid="760129">
              <a href="javascript:;">let us talk about love</a>
            </h3>

            <p>耀&nbsp;/&nbsp;1084次播放</p>
        </div>
        <span class="days">(上榜15天)</span>
        <span class="trend arrow-stay"> 0 </span>
    </li>

    <li class="clearfix">
        <span class="green-num-box">7</span>
        <a class="face" href="https://site.douban.com/lucifer/" target="_blank">
            <img src="https://img3.doubanio.com/view/site/small/public/b8e86e945f43312.jpg">
        </a>
        <div class="intro">
            <h3 class="icon-play" data-sid="760296">
              <a href="javascript:;">超人之歌</a>
            </h3>

            <p>王敖（音乐人）&nbsp;/&nbsp;63次播放</p>
        </div>
        <span class="days">(上榜1天)</span>
        <span class="trend arrow-up"> 14 </span>
    </li>

    <li class="clearfix">
        <span class="green-num-box">8</span>
        <a class="face" href="https://site.douban.com/shage/" target="_blank">
            <img src="https://img3.doubanio.com/view/site/small/public/ebefa04f89fd082.jpg">
        </a>
        <div class="intro">
            <h3 class="icon-play" data-sid="760158">
              <a href="javascript:;">小启示录（demo）</a>
            </h3>

            <p>郑荆沙&nbsp;/&nbsp;875次播放</p>
        </div>
        <span class="days">(上榜13天)</span>
        <span class="trend arrow-stay"> 0 </span>
    </li>

    <li class="clearfix">
        <span class="green-num-box">9</span>
        <a class="face" href="https://site.douban.com/seamoon/" target="_blank">
            <img src="https://img3.doubanio.com/view/site/small/public/7185157680e7633.jpg">
        </a>
        <div class="intro">
            <h3 class="icon-play" data-sid="760067">
              <a href="javascript:;">山风</a>
            </h3>

            <p>施文&nbsp;/&nbsp;1530次播放</p>
        </div>
        <span class="days">(上榜15天)</span>
        <span class="trend arrow-down"> 2 </span>
    </li>

    <li class="clearfix">
        <span class="green-num-box">10</span>
        <a class="face" href="https://site.douban.com/panicake/" target="_blank">
            <img src="https://img3.doubanio.com/view/site/small/public/d3ebde8c1fa6e72.jpg">
        </a>
        <div class="intro">
            <h3 class="icon-play" data-sid="759960">
              <a href="javascript:;">sad</a>
            </h3>

            <p>panicake&nbsp;/&nbsp;1385次播放</p>
        </div>
        <span class="days">(上榜15天)</span>
        <span class="trend arrow-down"> 5 </span>
    </li>


    <li class="clearfix">
        <span class="green-num-box">11</span>
        <div class="intro">
            <p class="icon-play" data-sid="760256">
                <a href="javascript:;">《他写的歌》</a>
                BAND STRANGER&nbsp;/&nbsp;19次播放
            </p>
        </div>
        <span class="days">(上榜1天)</span>
        <span class="trend arrow-up"> 10 </span>
    </li>

    <li class="clearfix">
        <span class="green-num-box">12</span>
        <div class="intro">
            <p class="icon-play" data-sid="760139">
                <a href="javascript:;">速食试炼</a>
                拟 白&nbsp;/&nbsp;1941次播放
            </p>
        </div>
        <span class="days">(上榜15天)</span>
        <span class="trend arrow-down"> 3 </span>
    </li>

    <li class="clearfix">
        <span class="green-num-box">13</span>
        <div class="intro">
            <p class="icon-play" data-sid="760279">
                <a href="javascript:;">说散就散</a>
                尼克福&nbsp;/&nbsp;116次播放
            </p>
        </div>
        <span class="days">(上榜3天)</span>
        <span class="trend arrow-up"> 7 </span>
    </li>

    <li class="clearfix">
        <span class="green-num-box">14</span>
        <div class="intro">
            <p class="icon-play" data-sid="760294">
                <a href="javascript:;">终于明白（800）</a>
                博友文化音乐工作室&nbsp;/&nbsp;13次播放
            </p>
        </div>
        <span class="days">(上榜1天)</span>
        <span class="trend arrow-up"> 7 </span>
    </li>

    <li class="clearfix">
        <span class="green-num-box">15</span>
        <div class="intro">
            <p class="icon-play" data-sid="760059">
                <a href="javascript:;">一件儿新睡衣（2020新版demo）</a>
                朱骏宁&nbsp;/&nbsp;233次播放
            </p>
        </div>
        <span class="days">(上榜6天)</span>
        <span class="trend arrow-up"> 1 </span>
    </li>

    <li class="clearfix">
        <span class="green-num-box">16</span>
        <div class="intro">
            <p class="icon-play" data-sid="760237">
                <a href="javascript:;">大风吹demo</a>
                宝丸&nbsp;/&nbsp;144次播放
            </p>
        </div>
        <span class="days">(上榜6天)</span>
        <span class="trend arrow-down"> 1 </span>
    </li>

    <li class="clearfix">
        <span class="green-num-box">17</span>
        <div class="intro">
            <p class="icon-play" data-sid="760007">
                <a href="javascript:;">Summer</a>
                Shadow喜欢柠檬&nbsp;/&nbsp;158次播放
            </p>
        </div>
        <span class="days">(上榜6天)</span>
        <span class="trend arrow-down"> 5 </span>
    </li>

    <li class="clearfix">
        <span class="green-num-box">18</span>
        <div class="intro">
            <p class="icon-play" data-sid="760144">
                <a href="javascript:;">欲生欲死欲悲搏欲辉煌陇爱企在家己ㄟ土地 Demo</a>
                六甲番&nbsp;/&nbsp;256次播放
            </p>
        </div>
        <span class="days">(上榜13天)</span>
        <span class="trend arrow-down"> 7 </span>
    </li>

    <li class="clearfix">
        <span class="green-num-box">19</span>
        <div class="intro">
            <p class="icon-play" data-sid="760141">
                <a href="javascript:;">三匹罐demo 20200811</a>
                小淞&nbsp;/&nbsp;123次播放
            </p>
        </div>
        <span class="days">(上榜3天)</span>
        <span class="trend arrow-down"> 1 </span>
    </li>

    <li class="clearfix">
        <span class="green-num-box">20</span>
        <div class="intro">
            <p class="icon-play" data-sid="760183">
                <a href="javascript:;">夕阳漫步（demo）</a>
                朱华东&nbsp;/&nbsp;209次播放
            </p>
        </div>
        <span class="days">(上榜10天)</span>
        <span class="trend arrow-down"> 7 </span>
    </li>
</ul>


"""

soup = BeautifulSoup(resp, 'html.parser')
# print(soup.find_all('li'))
nodes = soup.find_all('li')
for node in nodes:
    # msg = node.find_all('span', {'class': 'intro'})
    print(node.find('a', href='javascript:;').text)
    print(node.find('a', {'href': 'javascript:;'}).text)
