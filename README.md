# MyLifeWrapped
I tried to create something like Spotify Wrapped, but for all my life activities.

To achieve this, I did the next things:<br>
1. Get data from Google Calendar using google API and transfer it to table via pandas.
1. Get some stats from this data using pandas and sklearn.
1. Create unique plots with this stats using matplotlib.
1. Generate some beautiful backgrounds (which I don't share here)
via DALL-E-2 with prompt `colorful abstract art with gradients in pastel colors`.
1. Put text and plots with stats on these backgrounds and crop them using PIL.
1. Upload the images created to Google Drive and then insert them in Google Slides using google API.
1. Repeat this process every day automatically.

In the end I get something like this:
<p align="center">
<img src="mlw_demo.gif" width="300">
</p>

